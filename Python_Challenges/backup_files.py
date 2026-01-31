import shutil
import os

def backup_folder(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for file in os.listdir(source):
        src_path = os.path.join(source, file)
        dest_path = os.path.join(destination, file)

        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)

    print("Backup completed!")


if __name__ == "__main__":
    backup_folder("data", "backup_data")
