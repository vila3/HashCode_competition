import cPickle as pickle
from os import mkdir
from os.path import isfile, exists
import shutil

import numpy as np


def pickles_files_path(file_name, file_path=''):
    return "tmp/" + file_path + "/" + file_name + ".npz"


def main():
    print 'Start...'

    input_file = raw_input('Input file: ')
    # input_file = 'input/me_at_the_zoo.in'
    file_name = input_file.split('/')[-1].split('.')[0]

    pickle_files_path = {
        'infos': pickles_files_path("infos", file_path=file_name),
        'videos_sizes': pickles_files_path("videos_sizes", file_path=file_name),
        'endpoints_objects': pickles_files_path("endpoints_objects", file_path=file_name),
        'calculation_objects': pickles_files_path("calculation_objects", file_path=file_name),
    }

    f = open(input_file, 'r')
    f_list = f.readlines()

    first_line = f_list.pop(0)[:-1].split(' ')

    # ask if load data from caches files or from input files
    use_cached_data = (raw_input('Use cached data? (y/n): ') == 'y')
    if exists('tmp/' + file_name) and not use_cached_data:
        shutil.rmtree('tmp/' + file_name)

    if not exists('tmp/' + file_name):
        mkdir('tmp/' + file_name)

    if isfile(pickle_files_path['infos']):
        infos = pickle.load(open(pickle_files_path['infos'], "rb"))
    else:

        infos = {
            'n_videos': int(first_line[0]),
            'n_endpoints': int(first_line[1]),
            'n_request_descr': int(first_line[2]),
            'n_caches': int(first_line[3]),
            'caches_size': int(first_line[4])
        }

        pickle.dump(infos, open(pickle_files_path['infos'], "wb"))

    data = None
    table_ep_requests = None
    endpoints_latency_data_center = None
    table_ep_cchs = None

    # try to load data from caches files, if not caches files doesn't exists read data from input file
    try:
        videos_sizes = pickle.load(open(pickle_files_path['videos_sizes'], "rb"))

        data = np.load(pickle_files_path['endpoints_objects'])

        if not (
                'table_endpoints_requests' and 'endpoints_latency_data_center' and 'table_endpoints_caches') in data.keys():
            data.close()
            raise IOError

        endpoints_latency_data_center = data['endpoints_latency_data_center']
        table_ep_cchs = data['table_endpoints_caches']
        table_ep_requests = data['table_endpoints_requests']

        data.close()

        print 'Load from cache!'

    except IOError:
        print 'Data not in cache, prepare to read file...'

        # array with videos sizes
        # size -> (1D) #videos
        videos_sizes = map(int, f_list.pop(0)[:-1].split(' '))

        pickle.dump(videos_sizes, open(pickle_files_path['videos_sizes'], "wb"))

        # array with latencies from endpoints to data center
        # access with enpoint id
        # size -> (1D) #enpoints
        endpoints_latency_data_center = np.zeros(shape=infos['n_endpoints'])

        # table to relation endpoints latency to a specific cache
        # size -> (2D) lines=#endpoints | columns=#caches
        table_ep_cchs = np.zeros(shape=(infos['n_endpoints'], infos['n_caches']))

        # go to over all endpoints informations
        #   read latency from endpoint to datacenter and save on endpoints_latency_data_center
        #   read latency from that endpoint to cache and save on table_ep_cchs
        for i in range(0, infos['n_endpoints']):
            endpoint_info = f_list.pop(0)[:-1].split(' ')
            endpoints_latency_data_center[i] = int(endpoint_info[0])
            for j in range(0, int(endpoint_info[1])):
                cache = f_list.pop(0)[:-1].split(' ')
                table_ep_cchs[i, int(cache[0])] = int(cache[1])

        print "Reading...\n"

        # table to relation videos requests with enpoint from they come
        # size -> (2D) lines=#endpoints | columns=#videos
        table_ep_requests = np.zeros(shape=(infos['n_endpoints'], infos['n_videos']))

        # go over all request descriptions
        #   read request information
        #   verify if video from that request has a size greater than cache size
        #   if so -> save the #requests of that video from the specific endpoint on table_ep_requests
        for i in range(0, int(infos['n_request_descr'])):
            videos_info = f_list.pop(0)[:-1].split(' ')
            video_id = int(videos_info[0])
            if videos_sizes[video_id] > infos['caches_size']:
                continue
            table_ep_requests[int(videos_info[1]), int(videos_info[0])] = int(videos_info[2])

        # caches data
        np.savez(pickle_files_path['endpoints_objects'],
                 endpoints_latency_data_center=endpoints_latency_data_center,
                 table_endpoints_caches=table_ep_cchs,
                 table_endpoints_requests=table_ep_requests)

    print 'Data loaded!'

    if use_cached_data:
        data = np.load(pickle_files_path['calculation_objects'])
        matrix_caches_requests = data['matrix_caches_requests']
    else:
        matrix_caches_requests = np.zeros(shape=(infos['n_videos'], infos['n_caches']), dtype='int')

        total_latency_dataCenter_matrix = table_ep_requests * np.transpose(endpoints_latency_data_center)[:, None]

        print 'Start calculations...'

        for i in range(0, table_ep_requests.shape[1]):
            x = table_ep_requests[:, i]
            latency_dataCenter = total_latency_dataCenter_matrix[:, i]
            tmp_matrix = latency_dataCenter[:, None] - table_ep_cchs * x[:, None]
            matrix_caches_requests[i, :] = np.sum(tmp_matrix, axis=0)

        print 'Almost done...'
        np.savez(pickle_files_path['calculation_objects'], matrix_caches_requests=matrix_caches_requests)

    tmp_matrix = (-matrix_caches_requests).argsort(axis=None, kind='mergesort')
    tmp_matrix = np.unravel_index(tmp_matrix, matrix_caches_requests.shape)
    index_matrix_caches_requests_sorted = np.vstack(tmp_matrix).T

    del tmp_matrix
    del matrix_caches_requests

    import gc
    gc.collect()

    caches_ocup_size = np.zeros(infos['n_caches'])
    caches_videos_id = [[] for i in range(infos['n_caches'])]

    print 'Just some more calculations...'

    # print table_ep_requests_cpy
    while True:
        request_cache = index_matrix_caches_requests_sorted[0]
        if index_matrix_caches_requests_sorted.shape[0] <= 1:
            break
        index_matrix_caches_requests_sorted = index_matrix_caches_requests_sorted[1:]
        ep_of_cch = np.nonzero(table_ep_cchs[:, request_cache[1]])[0]
        # print ep_of_cch
        v_reqs = np.nonzero(table_ep_requests[:, request_cache[0]])[0]
        # print v_reqs
        v_reqs_to_rm = np.intersect1d(ep_of_cch, v_reqs, assume_unique=True)
        # print v_reqs_to_rm

        if v_reqs_to_rm.size <= 0:
            # print 'ok ;-)'
            continue
        # print table_ep_requests_cpy[:, request_cache[1]]
        if caches_ocup_size[request_cache[1]] + videos_sizes[request_cache[0]] <= infos['caches_size']:
            caches_ocup_size[request_cache[1]] += videos_sizes[request_cache[0]]
            (caches_videos_id[request_cache[1]]).append(request_cache[0])
            table_ep_requests[:, request_cache[0]][v_reqs_to_rm] = 0
            # print table_ep_requests_cpy[:, request_cache[1]]

    print 'Writting output...'

    f_out = open('output/' + file_name + '.out', 'w')
    f_out.write(str(len(caches_ocup_size)) + '\n')
    for i in range(0, len(caches_videos_id)):
        f_out.write(str(i))
        for videos_id in caches_videos_id[i]:
            f_out.write(' ' + str(videos_id))
        f_out.write('\n')


if __name__ == '__main__':
    main()

############  tests ##############

# a = np.array([1, 2, 4, 2,
#               3, 2, 1, 1])
# b = np.array([[1, 2],
#               [1, 0]])
# a[b[0, :]] = 0
# print a

#
# c = 15 * [3]
# print c
# f_out = open('output/1_output.out', 'w')
# f_out.write(str(123) + '\n')
# f_out.write(str(1) + ' ')

#
# matrix3d = np.zeros(shape=(2, 4), dtype='int')
# for i in range(0, b.shape[1]):
#     x = b[:, i]
#     x = np.transpose(x)
#     print x
#     print '--'
#     tmp_matrix = a * x[:, None]
#     print tmp_matrix
#     print '--'
#     matrix3d[i, :] = np.sum(tmp_matrix, axis=0)
#     print matrix3d
#     print '<-->'

# a = np.array([[7, 3, 0, 2],
#               [3, 0, 1, 7]])
#
# nonzeros = np.array(np.nonzero(a))
# print nonzeros
#
# print np.min(a[np.nonzero(a)])
#
# index = np.argmin(a[np.nonzero(a)])
# print index
#
# print '(', nonzeros[0, index], ',', nonzeros[1, index], ')'

#
# x = np.transpose(b[:, 0])
# print x[:, None] - a
# x = b[:, 0]
# print x[:, None] - a
