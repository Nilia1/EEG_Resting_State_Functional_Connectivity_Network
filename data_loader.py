"""
data_loader.py
Contains helper functions to recursively load and segment EEG data.
"""
import os
import mne
import numpy as np
import config

# =============================================================================
# Helper Function: Process a Single File
# =============================================================================
def segment_one_file(file_path, window_size=config.WINDOW_SIZE):
    """
    Reads a single EEGLAB file, finds the 'Eyes Closed' events, and creates epochs.
    """
    # We search for the TEXT label '4', not the number 4
    TARGET_EVENT_LABEL = config.TARGET_EVENT_LABEL 
    
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='error')
        
        # 1. Get events AND the ID mapping
        # 'event_id_map' is a dictionary like: {'4': 1, '99': 2}
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
        sfreq = raw.info['sfreq']
        
        # 2. Check if the label '4' actually exists in this file
        if TARGET_EVENT_LABEL not in event_id_map:
            print(f"Warning: File {os.path.basename(file_path)} does not contain label '{TARGET_EVENT_LABEL}'. Found keys: {list(event_id_map.keys())}")
            return None
            
        # 3. Get the dynamic ID for '4' in this specific file
        current_id = event_id_map[TARGET_EVENT_LABEL]
        
        new_events = []
        found_start = False
        
        # 4. Find the event using the dynamic ID
        for event in events:
            if event[2] == current_id:
                start_sample = event[0]
                found_start = True
                
                # Duration to analyze (safety margin)
                duration_sec = config.DURATION_SEC 
                
                # Generate triggers
                for time_offset in np.arange(0, duration_sec, window_size):
                    sample_offset = int(time_offset * sfreq)
                    # Use 'current_id' to maintain consistency
                    new_events.append([start_sample + sample_offset, 0, current_id])
                break 
        
        if not found_start:
             return None

        new_events = np.array(new_events).astype(int)
        
        # 5. Create Epochs
        epochs = mne.Epochs(
            raw,
            new_events,
            # Map the dynamic ID to a readable name
            event_id={'Eyes Closed': current_id}, 
            tmin=0,
            tmax=window_size,
            baseline=None,
            preload=True,
            reject=None,
            verbose='error'
        )
        
        return epochs

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


# =============================================================================
# Function 2: Process the Entire Folder (Recursive)
# =============================================================================
def load_lemon_dataset(folder_path):
    """
    Main function that searches a folder (and all its subfolders) for .set files.
    It filters out 'Eyes Open' files and returns a list of processed data for 'Eyes Closed'.
    """
    all_epochs = []     # List to store the results from all subjects
    found_files = []    # Temporary list to store paths of files we want to process

    # --- Step 1: recursively find files ---
    print(f"Scanning '{folder_path}' for files...")
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            # We want files that end with .set BUT NOT those ending with EO.set (Eyes Open)
            if filename.endswith('.set') and not filename.endswith('EO.set'):
                full_path = os.path.join(root, filename)
                found_files.append(full_path)
    found_files.sort()
    print(f"Found {len(found_files)} suitable .set files in subfolders. Starting processing...")
    
    # --- Step 2: Loop through found files ---
    for i, full_path in enumerate(found_files):
        print(f"Processing ({i+1}/{len(found_files)}): {os.path.basename(full_path)}")
        epoch_data = segment_one_file(full_path)
        
        if epoch_data is not None:
            all_epochs.append(epoch_data)
            
    # --- Step 3: Check results ---
    if len(all_epochs) == 0:
        print("No valid data found.")
        return None

    print("------------------------------------------------")
    print(f"Done! Successfully loaded {len(all_epochs)} subjects.")
    
    return all_epochs