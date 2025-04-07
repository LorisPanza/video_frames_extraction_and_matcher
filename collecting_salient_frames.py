'''
@author: Loris Panza
'''
from pathlib import Path
import cv2
import os
import argparse


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
    # Skip the specified number of frames
    #print(f"Skipping {n_frames} frames.")
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
    
    return frame



def choose_and_save_similar_frames(file_path, output_dir, folder_name, remove_contours, frames_difference, skip_interval_long, skip_interval_small):
    """
    Allows manual selection of similar frame pairs to save for matching tasks.
    Enables skipping multiple frames at once.

    Parameters:
        file_path (str): Path to the input video file.
        output_dir (str): Directory to save the frame pairs. 
        skip_interval (int): Large Number of frames to skip at once after pressing a key.
        skip_interval_small (int): Small Number of frames to skip at once after pressing a key.
    """
    # Creating folder
    os.makedirs(output_dir, exist_ok=True)
    print("Folder name: ")
    print(folder_name)
    
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    print(f"Processing video: {file_path}.")
    
    if not cap.isOpened():
        print("Error: Cannot open the file.")
        return
    
    frame_index = 0
    previous_frame = None
    previous_frame_bgr = None
    pair_count = 0


    while True:
        #iterating over frames
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret or frame is None:
            print("End of video or error reading frame.")
            break
        
        if frames_difference!=1 and frame_index!=0:
            print(f"Skipping {frames_difference} frames.")
            frame = skip_frames(cap, frames_difference)
        
        # remove the contours if specified
        if remove_contours:
            frame = remove_black_borders_auto(frame)

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Wait for user input to decide whether to save the frames
        print("\n")
        print("Press 's' to save this pair, or any other key to skip this pair.")
        print(f"Press 'x' to skip {skip_interval_long} frames.")
        print(f"Press 'c' to skip {skip_interval_small} frames.")
        print(f"Press 'z' to end the analysis on the current video.")
        print(f"Frame index: {frame_index}")
        print(f"Pair saved until now: {pair_count}")
        print("\n")

        
        if previous_frame_bgr is not None:

            cv2.imshow("Previous Frame", previous_frame_bgr)
            cv2.imshow("Current Frame", frame)

            assert previous_frame_bgr.mean() != frame.mean()
    
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                # Save the frames as a pair
                frame_1_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_1.jpg")
                frame_2_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_2.jpg")
                cv2.imwrite(frame_1_path, previous_frame_bgr)
                cv2.imwrite(frame_2_path, frame)
                print(f"Saved pair: {frame_1_path} and {frame_2_path}")
                pair_count += 1
                frame_index += 1
            
            elif key == ord('x'):
                # Skip the specified number of frames
                frame = skip_frames(cap, skip_interval_long)
                if frame is None:
                    print("Skipped to the video end.")
                    break
                if remove_contours:
                    frame = remove_black_borders_auto(frame)
                previous_frame_bgr = frame
                frame_index += 1
                # Skip the rest of the loop and move on to the next frame, starts at the beginning on the for while the previous frame has been saved here
                continue
            
            elif key == ord('c'):
                # Skip the specified number of frames
                frame = skip_frames(cap, skip_interval_small)
                if frame is None:
                    print("Skipped to the video end.")
                    break
                if remove_contours:
                    frame = remove_black_borders_auto(frame)
                previous_frame_bgr = frame
                frame_index += 1
                # Skip the rest of the loop and move on to the next frame,  starts at the beginning on the for while the previous frame has been saved here
                continue
            elif key == ord("z"):
                # Interrupt the video
                break

        # Update the previous frame
        previous_frame_bgr = frame
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Finished processing. Total pairs saved: {pair_count}")
    if(len(os.listdir(output_dir))==0):
        # removing empty folder
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
    parser.add_argument("--frames_difference", type=int, default=1)
    parser.add_argument("--skip_interval_long", type=int, default=20, help="Number of frames to skip at once after pressing a key.")
    parser.add_argument("--skip_interval_small", type=int, default=3, help="Number of frames to skip at once after pressing a key.")

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
                                       subfix_frames, args.remove_contours, args.frames_difference ,args.skip_interval_long, args.skip_interval_small)
    
