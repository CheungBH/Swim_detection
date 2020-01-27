import os

main_folder = "processor/test"


def del_mid(folder):
    for file in os.listdir(folder):
        if "_" in file:
            os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    for f in os.listdir(main_folder):
        if os.path.isdir(os.path.join(main_folder, f)):
            del_mid(os.path.join(main_folder, f))
