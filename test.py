import os, sys


def getAllFiles(root,fileext):
    # print ('root: ', )
    with open ('tests/files.txt', 'w') as f:

        filepaths = []
        for base, dirs, files in os.walk(root):
            # print ('base, dirs: ', base, dirs)
            for file in files:
                if file.endswith(fileext):
                    # print (filepaths)
                    f.write(os.path.join(base, file) + '\n')
    

getAllFiles('swdata', 'java')