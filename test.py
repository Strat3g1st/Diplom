import os


def list_files_in_dir(dirname):
    dirfiles = os.listdir(dirname)
    result = []
    for filename in dirfiles:
        result.append(dirname + filename)
    return result


print(list_files_in_dir("C:\\Users\\maste\\Documents\\images\\"))
