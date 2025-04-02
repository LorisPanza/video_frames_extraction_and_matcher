from pathlib import Path
import cv2
import os
import argparse


def remove_black_borders_auto(frame):
    """
    Rimuove automaticamente i bordi neri da un frame, rilevando contorni validi.
    
    Parameters:
        frame (numpy.ndarray): Il frame da processare.
        
    Returns:
        numpy.ndarray: Frame ritagliato.
    """
    assert frame is not None
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
    Legge un video e mostra i frame uno alla volta.
    
    Parameters:
        file_path (str): Percorso del file video.
        
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


def choose_and_save_similar_frames(file_path, output_dir, folder_name, remove_contours, skip_interval_long=20, skip_interval_small=3):
    """
    Allows manual selection of similar frame pairs to save for matching tasks.
    Enables skipping multiple frames at once.

    Parameters:
        file_path (str): Path to the input video file.
        output_dir (str): Directory to save the frame pairs. 
        skip_interval (int): Large Number of frames to skip at once after pressing a key.
        skip_interval_small (int): Small Number of frames to skip at once after pressing a key.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    #folder_name = os.path.split(output_dir)[-1]
    print("Folder name: ")
    #print(os.path.split(output_dir))
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

    # Wait for user input to decide whether to save the frames
    print("Press 's' to save this pair, or any other key to skip this pair.")
    print(f"Press 'x' to skip {skip_interval_long} frames.")
    print(f"Press 'c' to skip {skip_interval_small} frames.")
    print(f"Press 'z' to end the analysis on the current video.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        if remove_contours:
            frame = remove_black_borders_auto(frame)

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if previous_frame is not None:

            cv2.imshow("Previous Frame", previous_frame_bgr)
            cv2.imshow("Current Frame", frame)
    
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                # Save the frames as a pair
                frame_1_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_1.jpg")
                frame_2_path = os.path.join(output_dir, f"{folder_name}_pair_{pair_count}_2.jpg")
                cv2.imwrite(frame_1_path, previous_frame_bgr)
                cv2.imwrite(frame_2_path, frame)
                print(f"Saved pair: {frame_1_path} and {frame_2_path}")
                pair_count += 1
            
            elif key == ord('x'):
                # Skip the specified number of frames
                print(f"Skipping {skip_interval_long} frames.")
                for _ in range(skip_interval_long):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if remove_contours:
                        frame = remove_black_borders_auto(frame)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    previous_frame = gray_frame
                    previous_frame_bgr = frame
                    frame_index += 1
                # Skip the rest of the loop and move on to the next frame
                continue
            elif key == ord('c'):
                # Skip the specified number of frames
                print(f"Skipping {skip_interval_small} frames.")
                for _ in range(skip_interval_small):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if remove_contours:
                        frame = remove_black_borders_auto(frame)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    previous_frame = gray_frame
                    previous_frame_bgr = frame
                    frame_index += 1
                # Skip the rest of the loop and move on to the next frame
                continue
            elif key == ord("z"):
                # Interrupt the video
                break

        # Update the previous frame
        previous_frame = gray_frame
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
        help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default="frames_source", help="path where outputs are saved")
    parser.add_argument("--remove_contours", type=bool, default='True', help="Set True if you want to automatically black bourders.")
    parser.add_argument("--name_folder", type=str, help="Set True if you want to automatically black bourders.")
    parser.add_argument("--gt", type=bool, default=False)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # avi and mp4 extension in dir
    avi_files = [f for f in os.listdir(args.input) if f.endswith('.avi') or f.endswith('.mp4')]
    avi_files_path = [os.path.join(args.input,f) for f in avi_files]

    for file in avi_files_path:
        video_name = Path(file).stem
        if args.name_folder:
            video_name = args.name_folder
        if args.gt:
            name_model = "gt"
        reading_video(file)
        choose_and_save_similar_frames(file, f"{args.out_dir}/salient_frames_{video_name}/{name_model}", video_name, args.remove_contours)
    
    #folder_vect = [f for f in os.listdir(args.out_dir)]
    #folder_vect_path = [os.path.join(args.out_dir,f) for f in folder_vect]

    #print(folder_vect_path)