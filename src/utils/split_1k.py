import argparse
import os
import sys
import errno


def open_file(output, counter):
    filename = "%s/%s/wiki_%s_sym" % (output, counter/10000, counter % 10000)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return open(filename, "w")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_list', type=str, help="Input data list."
                        " and clean up.", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output files."
                        " and clean up.", required=True)

    args = parser.parse_args()
    data_path = args.data_list
    output = args.output

    files = open(data_path).read().split()

    i = 1
    counter = 0
    max = 35
    current_file = open_file(output, counter)
    for f in files:
        current_data = open(f).read().split()
        for w in current_data:
            if i > max:
                current_file.close()
                counter += 1
                i = 1
                current_file = open("%s/wiki_%s_sym" % (output, counter), "w")
            if i != 1:
                current_file.write(" ")
            current_file.write(w)
            i += 1

    current_file.close()