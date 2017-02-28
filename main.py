import cPickle as pickle

import numpy as np
from os import mkdir
from os.path import isfile, exists

import sys


def pickles_files_path(file_name, file_path=''):
    return "tmp/" + file_path + "/" + file_name + ".npz"


def main():
    print 'Start...'

    # input_file = raw_input('Input file: ')
    input_file = 'input/kittens.in'
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

    if not exists('tmp/'+file_name):
        mkdir('tmp/'+file_name)

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
    table_endpoints_requests = None
    endpoints_latency_data_center = None
    table_endpoints_caches = None
    try:
        videos_sizes = pickle.load(open(pickle_files_path['videos_sizes'], "rb"))

        data = np.load(pickle_files_path['endpoints_objects'])

        if not (('table_endpoints_requests' and 'endpoints_latency_data_center' and 'table_endpoints_caches') in data.keys()):
            endpoints_latency_data_center = data['endpoints_latency_data_center']
            table_endpoints_caches = data['table_endpoints_caches']
            table_endpoints_requests = data['table_endpoints_requests']

            data.close()
            raise IOError
        data.close()

        print 'Load from cache!'

    except IOError:
        print 'Data not in cache, prepare to read file...'

        videos_sizes = map(int, f_list.pop(0)[:-1].split(' '))

        pickle.dump(videos_sizes, open(pickle_files_path['videos_sizes'], "wb"))

        endpoints_latency_data_center = np.zeros(shape=infos['n_endpoints'])

        table_endpoints_caches = np.zeros(shape=(infos['n_endpoints'], infos['n_caches']))

        for i in range(0, infos['n_endpoints']):
            endpoint_info = f_list.pop(0)[:-1].split(' ')
            endpoints_latency_data_center[i] = int(endpoint_info[0])
            for j in range(0, int(endpoint_info[1])):
                cache = f_list.pop(0)[:-1].split(' ')
                table_endpoints_caches[i, int(cache[0])] = int(cache[1])

        print "Reading...\n"

        table_endpoints_requests = np.zeros(shape=(infos['n_endpoints'], infos['n_videos']))

        for i in range(0, int(infos['n_request_descr'])):
            videos_info = f_list.pop(0)[:-1].split(' ')
            video_id = int(videos_info[0])
            if videos_sizes[video_id] > infos['caches_size']:
                continue
            table_endpoints_requests[int(videos_info[1]), int(videos_info[0])] = int(videos_info[2])

        np.savez(pickle_files_path['endpoints_objects'],
                 endpoints_latency_data_center=endpoints_latency_data_center,
                 table_endpoints_caches=table_endpoints_caches,
                 table_endpoints_requests=table_endpoints_requests)

    print 'Data loaded!'

    if data is not None and raw_input('Use cached data? (y/n) ') == 'y':
        data = np.load(pickle_files_path['calculation_objects'])
        matrix_caches_requests = data['matrix_caches_requests']
    else:
        matrix_caches_requests = np.zeros(shape=(infos['n_videos'], infos['n_caches']), dtype='int')

        total_latency_dataCenter_matrix = table_endpoints_requests * np.transpose(endpoints_latency_data_center)[:, None]

        print 'Start calculations...'

        for i in range(0, table_endpoints_requests.shape[1]):
            x = table_endpoints_requests[:, i]
            latency_dataCenter = total_latency_dataCenter_matrix[:, i]
            tmp_matrix = latency_dataCenter[:, None] - table_endpoints_caches * x[:, None]
            matrix_caches_requests[i, :] = np.sum(tmp_matrix, axis=0)

        print 'Almost done...'
        np.savez(pickle_files_path['calculation_objects'], matrix_caches_requests=matrix_caches_requests)

    non_zeros_index = np.array(np.nonzero(matrix_caches_requests))
    min_index_non_zero = np.argmin(matrix_caches_requests[np.nonzero(matrix_caches_requests)])
    min_index = (non_zeros_index[0, min_index_non_zero], non_zeros_index[1, min_index_non_zero])
    min_value = matrix_caches_requests[min_index]
    max_index = np.unravel_index(np.argmax(matrix_caches_requests), matrix_caches_requests.shape)
    max_value = matrix_caches_requests[max_index]

    print 'All done!'
    print
    print 'Min value:', min_value, '(%d, %d)' % min_index
    print 'Max value:', max_value, '(%d, %d)' % max_index

    #TODO rankear conjunto video-cache e atribui-los
    #TODO fazer output

if __name__ == '__main__':
    main()

############  tests ##############

a = np.array([[1, 2, 4, 2],
              [3, 2, 1, 1]])
b = np.array([[2, 3],
              [1, 2]])
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

# a = np.array([[1, 2, 4, 2],
#               [3, 2, 1, 1]])
# b = np.array([[2, 3],
#               [1, 2]])
#
# x = np.transpose(b[:, 0])
# print x[:, None] - a
# x = b[:, 0]
# print x[:, None] - a
