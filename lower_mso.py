if configs.sample_generation_mode == 'serial':
 

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom

        batches = int(np.ceil(configs.n_samples/configs.sample_batch_size))
        #n_samples = sample_batch_size * batches
        sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample y dim'], dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] , sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * 1, sample_x_padded + 2 * dataDims['conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            # if configs.do_conditioning: # assign conditions so the model knows what we want
            #     for i in range(len(configs.generation_conditions)):
            #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']

            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            #generator.train(False)
            model.eval()
            with torch.no_grad():  # we will not be updating weights
                for i in tqdm.tqdm(range(dataDims['conv field'] + 1, sample_y_padded + dataDims['conv field'] + 1)):  # for each pixel
                    for j in range(dataDims['conv field'], sample_x_padded + dataDims['conv field']):
                        for k in range(dataDims['channels']): # should only ever be 1
                            #out = generator(sample_batch.float())
                            out = model(sample_batch[:, :, i - dataDims['conv field'] - 1:i  + 1, j - dataDims['conv field']:j + dataDims['conv field'] + 1].float())
                            out = torch.reshape(out, (out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-2], out.shape[-1]))
                            probs = F.softmax(out[:, 1:, k, -1, dataDims['conv field']]/configs.softmax_temp, dim=1).data # the remove the lowest element (boundary)
                            sample_batch[:, k, i, j] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / dataDims['classes']  # convert output back to training space

                            del out, probs

            for k in range(dataDims['channels']):
                sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 1:, (configs.boundary_layers + 1) * dataDims['conv field']:-((configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space

   




            np.save('samples/sample-{}-temp-{}'.format(configs.run_num, configs.softmax_temp), sample.cpu())

    
elif configs.sample_generation_mode == 'parallel':
    #    if configs.CUDA:
    #        cuda.synchronize()
     #   time_ge = time.time()
        dataDims['sample x dim']= dataDims['sample x dim']*configs.sample_outpaint_ratio
        dataDims['sample y dim']=dataDims['sample y dim']*configs.sample_outpaint_ratio


        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom


        sample = torch.ByteTensor(configs.n_samples, dataDims['channels'], dataDims['sample y dim'], dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for image in range(configs.n_samples):  # can't do these all at once so we do it in batches
            print('Image {} of {} images'.format(image + 1, configs.n_samples))
            sample_batch = torch.FloatTensor(1, dataDims['channels'] , sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * 1, sample_x_padded + 2 * dataDims['conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            # if configs.do_conditioning: # assign conditions so the model knows what we want
            #     for i in range(len(configs.generation_conditions)):
            #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']

            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            '''
            the key to this speedup is that workers which are separated by at least conv_field CAN work in parallel on different rows
            we can have a maximum of samble_batch_size workers, as workers reach j > conv_field, the next row (i+1) becomes available for another worker
            we will distribute workers as if they were all working on separate images, even though they are just working on different parts of the same image
            a list of rows which are available to start working on, and assign workers based on capacity
            this is all accomplished by recasting 'sample_batch' at each iteration (j) of the generator
            '''

            model.train(False)
            model.eval()
            with torch.no_grad():  # we will not be updating weights
                finished_rows = 0
                available_rows = [dataDims['conv field'] + 1]
                active_rows = []
                available_workers = configs.sample_batch_size
                row_indices = (np.zeros(sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * (1-1)) + dataDims['conv field']).astype(int)
                initialized = 0
                # record = []
                pbar = barthing(total=sample_y_padded)
                while finished_rows < (sample_y_padded):  # generate row-by-row
                    # check if we have spare rows and spare workers
                    if initialized == 0:  # first row - initialization
                        # initialize a row
                        row = available_rows[0]
                        sample_bundle = sample_batch[:, :, row - dataDims['conv field']  - 1:row +dataDims['conv field'] * (1-1) + 1, int(row_indices[row] - dataDims['conv field']):int(row_indices[row] + dataDims['conv field'] + 1)] * 1

                        if configs.CUDA:
                            sample_bundle = sample_bundle.cuda()

                        available_rows = available_rows[1:]  # eliminate first element
                        active_rows.append(row)
                        available_workers -= 1
                        initialized = 1

                    elif (available_rows != []) and (available_workers > 0):
                        if active_rows[-1] < (sample_y_padded + dataDims['conv field'] - 1 + 1):  # unless we are already on the final row
                            # initialize a row
                            row = available_rows[0]
                            sample_bundle = torch.cat((sample_bundle, sample_batch[:, :, row - dataDims['conv field'] - 1:row + dataDims['conv field'] *(1-1) + 1, int(row_indices[row] - dataDims['conv field']):int(row_indices[row] + dataDims['conv field'] + 1)]) * 1, 0)
                            active_rows.append(row)
                            available_rows = available_rows[1:]  # eliminate first element
                            available_workers -= 1

                    for k in range(dataDims['channels']):  # actually do the generation
                        out = model(sample_bundle.float())  # query the network about only area within the receptive field
                        out = torch.reshape(out, (out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-2], out.shape[-1]))  # reshape to select channels
                       # print(out[:, 1:, k, -1, dataDims['conv field']])
                        probs = F.softmax(out[:, 1:, k, -1, dataDims['conv field']] / configs.softmax_temp, dim=1).data  # the remove the lowest element (boundary)
                        logits = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / dataDims['classes']
                        for dep in range(sample_bundle.shape[0]):  # assign the new outputs in the right spot
                            sample_batch[:, k, active_rows[dep], row_indices[active_rows[dep]]] = logits[dep].data

                        # record.append(sample_batch[0,0,:,:].cpu().detach().numpy() * 1)

                    # check if we finished a row
                    if row_indices[active_rows[0]] == (sample_x_padded + dataDims['conv field'] - 1):
                        # only one should be possible at a time
                        # delete this row from active list and add to the finished rows
                        active_rows = active_rows[1:]  # the earliest row must be the one which has finished
                        sample_bundle = sample_bundle[1:, :, :, :]
                        finished_rows += 1
                        pbar.update(1)
                        available_workers += 1  # free up a worker

                    # check if any rows have been freed up
                    if active_rows != []:
                        if row_indices[active_rows[-1]] == (2 * dataDims['conv field'] + 1):  # if the bottom worker is more than dataDims['conv field'] from the bound (which has dataDims['conv field'] added as padding), this row comes available
                            # in fact, when working with a blind spot, we can do even better - initial_filter_size + 1 is enough
                            available_rows.append(active_rows[-1] + 1)

                    # update sample_bundle by one pixel to the right
                    for dep in range(sample_bundle.shape[0]):
                        row = active_rows[dep]
                        # update sample bundle_
                        sample_bundle[dep, :, :, :] = sample_batch[:, :, row - dataDims['conv field'] - 1: row + dataDims['conv field'] * ( 1- 1) + 1, int(row_indices[row] - dataDims['conv field'] + 1):int(row_indices[row] + dataDims['conv field'] + 1 + 1)]
                        # update row indices
                        row_indices[row] += 1

                pbar.close()

            for k in range(dataDims['channels']):
                sample[image, k, :, :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 1:, (configs.boundary_layers + 1) * dataDims['conv field']:-((configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space, +1 in y dim to get rid of first row


   




           
            np.save('samples/sample-{}-temp-{}'.format(configs.run_num, configs.softmax_temp), sample.cpu())



   
