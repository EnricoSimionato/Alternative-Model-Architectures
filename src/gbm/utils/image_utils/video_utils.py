import os
import sys
import cv2
import re


def extract_number(filename: str) -> int:
    """
    Extracts the first number found in the filename.

    Args:
        filename (str): The name of the file.

    Returns:
        int: The extracted number.
    """
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0


def create_video_from_images(
        path_to_images: str,
        filter_string: str = "",
        path_to_store_video: str = None,
        video_name: str = "video.mp4",
        fps: int = 1
) -> None:
    """
    Create a video from a sequence of images that contain the filter string.

    Args:
        path_to_images (str):
            The path to the directory containing the images.
        filter_string (str, optional):
            The string to filter images. Defaults to "" (no filter).
        path_to_store_video (str, optional):
            The path to store the video. Defaults to None.
        video_name (str, optional):
            The name of the video file. Defaults to "video.mp4".
        fps (int, optional):
            The frames per second of the video. Defaults to 1.
    """

    if not os.path.exists(path_to_images) or not os.path.isdir(path_to_images):
        raise FileNotFoundError(f"Directory {path_to_images} not found")

    if path_to_store_video is not None and (
            not os.path.exists(path_to_store_video) or not os.path.isdir(path_to_store_video)):
        os.makedirs(path_to_store_video)

    if path_to_store_video is None:
        path_to_store_video = os.path.join(path_to_images, video_name)
    else:
        path_to_store_video = os.path.join(path_to_store_video, video_name)

    images = [img for img in os.listdir(path_to_images) if
              img.endswith(".png") and (filter_string in img if filter_string else True)]

    # Sort images by the number extracted from their filenames
    images.sort(key=extract_number)

    if not images:
        raise FileNotFoundError(f"No images found with filter string '{filter_string}' in directory {path_to_images}")

    frame = cv2.imread(os.path.join(path_to_images, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path_to_store_video, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(path_to_images, image)))

    cv2.destroyAllWindows()
    video.release()


def get_src_path():
    current_directory = os.getcwd()
    src_path = os.path.join(current_directory.split('src')[0], 'src')
    return src_path


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python video_generator.py <experiment_name> [filter_string]")
        sys.exit(1)

    experiment_name = sys.argv[1]
    filter_string = sys.argv[2] if len(sys.argv) == 3 else ""
    src_path = get_src_path()
    path_to_experiment = os.path.join(src_path, 'experiments', 'performed_experiments', experiment_name, 'images')

    print(f"Creating video from images in: '{path_to_experiment}' with filter string: '{filter_string}'")

    if not os.path.exists(path_to_experiment):
        print(f"Image directory '{path_to_experiment}' not found.")
        sys.exit(1)

    create_video_from_images(path_to_images=path_to_experiment, filter_string=filter_string)


if __name__ == "__main__":
    main()
