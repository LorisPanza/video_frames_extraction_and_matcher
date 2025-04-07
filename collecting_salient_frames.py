'''
@author: Loris Panza
'''
from pathlib import Path
import cv2
import os
import argparse
from matching.utils import pair_images_in_folder

def remove_black_borders_auto(frame):
    """
    Remove black contours from the frames capture.
    
    Parameters:
        frame (numpy.ndarray): Frame to process.
        
    Returns:
        numpy.ndarray: Frame without black borders.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Trova i contorni che delimitano l'area non nera
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))  # Contorno più grande
        return frame[y:y+h, x:x+w]
    return frame  # Se nessun contorno, ritorna il frame originale


def reading_video(file_path):
    """
    Play a video and allow to save pair of subsequent frames.
    
    Parameters:
        file_path (str): Video path.
        
    Returns:
        None
    """
    cap = cv2.VideoCapture(file_path)
    print(f"Capturing video in {file_path}")
    if not cap.isOpened():
        print("Error: Cannot open the file.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Frame", frame)
            cv2.waitKey(2)

    cap.release()
    cv2.destroyAllWindows()


def skip_frames(cap, n_frames):
    """
    Skip a specified number of frames in a video capture.
    
    Parameters:
        cap: Pointer to the video capture object.
        n_frames (int): Number of frames to skip.
        
    Returns:
        frame (numpy.ndarray): The last frame read after skipping.
        cap: Pointer to the video capture object.
    """
    print(f"Skipping {n_frames} frames.")
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
    
    return frame, cap



def choose_and_save_similar_frames(file_path, output_dir, folder_name, remove_contours, frames_difference, skip_interval_long, skip_interval_small):
        """
        Allows manual selection of similar frame pairs to save for matching tasks.
        Enables skipping multiple frames at once.

        Parameters:
            file_path (str): Path to the input video file.
            output_dir (str): Directory to save the frame pairs.
            folder_name (str): Name to be used in saved frame file names.
            remove_contours (bool): Whether to remove borders from the frames.
            frames_difference (int): Number of frames to skip forward to form a pair.
            skip_interval_long (int): Number of frames to skip with key 'x'.
            skip_interval_small (int): Number of frames to skip with key 'c'.
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print("Folder name: ")
        print(folder_name)
        
        # Open the video file for reading
        cap = cv2.VideoCapture(file_path)
        print(f"Processing video: {file_path}.")
        
        if not cap.isOpened():
            print("Error: Cannot open the file.")
            return

        frame_index = 0               # Counter for current frame
        pair_count = 0               # Counter for saved frame pairs

        while True:
            # Read next frame
            ret, actual_frame = cap.read()
            
            # Stop if we reach end of video or can't read the frame
            if not ret or actual_frame is None:
                print("End of video or error reading frame.")
                break

            # Try to get the "future" frame by skipping ahead a few frames
            if frames_difference != 1:
                print("Skipping in future")
                future_frame, future_cap = skip_frames(cap, frames_difference)
            
            # Optionally remove borders from both frames
            if remove_contours:
                actual_frame = remove_black_borders_auto(actual_frame)
                future_frame = remove_black_borders_auto(future_frame)
            
            # Display controls to the user
            print("\n")
            print("Press 's' to save this pair, or any other key to skip this pair.")
            print(f"Press 'x' to skip {skip_interval_long} frames.")
            print(f"Press 'c' to skip {skip_interval_small} frames.")
            print(f"Press 'z' to end the analysis on the current video.")
            print(f"Frame index: {frame_index}")
            print(f"Pair saved until now: {pair_count}")
            print("\n")

            if future_frame is not None:
                # Show the current and future frames
                cv2.imshow("Current Frame", actual_frame)
                cv2.imshow("Future Frame", future_frame)

                # Sanity check: make sure the two frames are not identical
                assert future_frame.mean() != actual_frame.mean()

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('s'):
                    # Save both frames as a pair of images
                    frame_1_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_1.jpg")
                    frame_2_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_2.jpg")
                    cv2.imwrite(frame_1_path, actual_frame)
                    cv2.imwrite(frame_2_path, future_frame)
                    print(f"Saved pair: {frame_1_path} and {frame_2_path}")
                    pair_count += 1
                    frame_index += 1

                elif key == ord('x'):
                    # Skip a large number of frames
                    print("Skipping from actual")
                    _, cap = skip_frames(cap, skip_interval_long)
                    frame_index += 1
                    continue  # Skip the rest of the loop

                elif key == ord('c'):
                    # Skip a small number of frames
                    print("Skipping from actual")
                    _, cap = skip_frames(cap, skip_interval_small)
                    frame_index += 1
                    continue  # Skip the rest of the loop

                elif key == ord("z"):
                    # Stop the frame browsing loop
                    break

            frame_index += 1

        # Release video and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        print(f"Finished processing. Total pairs saved: {pair_count}")

        # Clean up if no frames were saved
        if len(os.listdir(output_dir)) == 0:
            os.rmdir(output_dir)
        
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video salient frames extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
  
    parser.add_argument(
        "--input",
        type=str,
        default="D:/dataset/superresolution/videos",
        help="path to either dir with dirs with videos",
    )
    parser.add_argument("--out_dir", type=Path, default="frames_source", help="path where outputs are saved")
    parser.add_argument("--remove_contours", type=bool, default='True', help="Set True if you want to automatically black bourders.")
    parser.add_argument("--inner_folder_name", type=str, default="frames", help = "subdirectory name")
    parser.add_argument("--frames_difference", type=int, default=1, help = "Number of frames to skip forward to form a pair.")
    parser.add_argument("--skip_interval_long", type=int, default=20, help="Number of frames to skip at once after pressing a key.")
    parser.add_argument("--skip_interval_small", type=int, default=3, help="Number of frames to skip at once after pressing a key.")
    parser.add_argument("--images_to_be_paired", type=bool, default=True, help ="pair the image in folders by the name.") # frames_source\salient_frames_name_folder that could be salient_frames_name_folder/models/list_of_images.png or salient_frames_name_folder/models/image_folders/list_of_images.png


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # avi and mp4 extension in dir
    avi_files = [f for f in os.listdir(args.input) if f.endswith('.avi') or f.endswith('.mp4')]
    avi_files_path = [os.path.join(args.input,f) for f in avi_files]

    for file in avi_files_path:
        subfix_frames = Path(file).stem
        #reading_video(file)
        choose_and_save_similar_frames(file, f"{args.out_dir}/salient_frames_{subfix_frames}/{args.inner_folder_name}", 
                                       subfix_frames, args.remove_contours, args.frames_difference, args.skip_interval_long, args.skip_interval_small)
    
    if(not(args.images_to_be_paired is None)):
            print("Creating folder")
            # args.out to save matched frames
            assert not(args.out_dir is None)
            pair_images_in_folder(args.out_dir, args.out_dir)
    
