import glob
import PySimpleGUI as sg 
import os

from PIL import Image, ImageTk

import pandas as pd

import numpy as np

from IPython.display import display

import time

def load_annotated_csv(folder):
    df_annotated = None
    if os.path.isfile("annotated_data.csv"):
        df_annotated = pd.read_csv("annotated_data.csv")
    else:
        folders = []
        names = []
        labels = []

        d = {
            "Folder": folders, 
            "Name_of_Image": names,
            "Label":labels}

        df_annotated = pd.DataFrame(d)
    display(df_annotated)
    return df_annotated

def load_excluded_csv():
    df_excluded = None
    if os.path.isfile("excluded_data.csv"):
        df_excluded = pd.read_csv("excluded_data.csv")
    else:
        folders = []
        names = []
        labels = []

        d = {
            "Folder": folders, 
            "Name_of_Image": names}

        df_excluded = pd.DataFrame(d)
    display(df_excluded)
    return df_excluded

def handle_annotation_edition(df_annotated, name, folder, label):
    folders = df_annotated["Folder"].to_numpy()
    names = df_annotated["Name_of_Image"].to_numpy()
    labels = df_annotated["Label"].to_numpy()

    for i in range(len(names)):
        if folders[i] == folder: 
            if names[i] == name:
                labels[i] = label

    d = {
        "Folder": folders, 
        "Name_of_Image":names, 
        "Label":labels
    }
    df_annotated = pd.DataFrame(d)
    return df_annotated


def add_annotated_record(folder, name, label, df_annotated, window):
    
    if label != "":
        if check_annotation_edition(df_annotated, name, folder, window):
            df_annotated = handle_annotation_edition(df_annotated, name, folder, label)
        else: 
            d_aux = {
                "Folder": [folder],
                "Name_of_Image":[name],
                "Label":[label]
            }

            df_aux = pd.DataFrame(d_aux)
    
            df_annotated = df_annotated.append(df_aux)

            display(df_annotated)

    return df_annotated

def save_annotated_csv(df_annotated):
    if df_annotated.empty == False:
        d = {
            "Folder": df_annotated["Folder"].to_numpy(),
            "Name_of_Image": df_annotated["Name_of_Image"].to_numpy(), 
            "Label": df_annotated["Label"].to_numpy()
        }

        df_annotated = pd.DataFrame(d)
        display(df_annotated)
        df_annotated.to_csv("annotated_data.csv", index = False)

def save_excluded_csv(df_excluded):
    if df_excluded.empty == False:
        d = {
            "Folder": df_excluded["Folder"].to_numpy(),
            "Name_of_Image": df_excluded["Name_of_Image"].to_numpy()
        }

        df_excluded = pd.DataFrame(d)
        display(df_excluded)
    df_excluded.to_csv("excluded_data.csv", index = False)


def parse_folder(path):
    images = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png')
    return images

def load_image(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data = photo_img)
    except:
        print("Unable to open ", path)


left_file_list_column = [
    [
        sg.Input(size = (25, 1), enable_events = True, key = "folder"),
        sg.FolderBrowse(key = "folder_browser"),
    ],
    [
        sg.Listbox(
            values = [], enable_events = True, size = (40, 20),
            key = "file_list"
        )
    ],
]

image_viewer_column = [
    [
        sg.Text("Choose an image from the list on the left or click next or prev buttons.", key = "text1", visible = False),

    ],
    [
        sg.Image(key = "image"),
    ],
    [
       sg.Text("You have already set the annotation of this image. Do you want to edit it? ", key = "edit_text", visible = False),
       sg.Text("You have set wrong the bounding box of this image. Do you want to edit it? ", key = "wrong_bbox_text", visible = False),
       sg.Button("Edit", visible = False),
       sg.Text("Is the bounding box correctly placed? ", visible = False, key = "bbox_text"),
       sg.Radio("Yes", "radio1", key = "Yes", visible = False),
       sg.Radio("No","radio1", key = "No", visible = False),
       sg.Button("Confirm", visible = False)
    ],
    [
        sg.Text("We have annotated that this bounding box is wrong. ", visible = False, key = "no_bbox"),
        sg.Text("Please write the name of the fish or coral that appears within the bounding box", visible = False, key = "name_text")
    ],
    [
        sg.Input(size = (25, 1), enable_events = True, key = "label", background_color = 'white', visible = False)
    ],
    [
        sg.Button("Previous Image", visible = False, key = "Prev"),
        sg.Button("Next Image", visible = False, key = "Next"),
        sg.Button("Exit and see location of annotations", key = "Exit"),
        sg.Text(key = "final_text"),
        sg.Button("Quit", key = "Quit", visible = False), 
        sg.Button("No Quit", visible = False)
    ] 
]


layout = [
    [
        sg.Column(left_file_list_column),
        sg.VSeperator(), 
        sg.Column(image_viewer_column),

    ]
]

def show_bbox_part(window):
    window["bbox_text"].update(visible = True)
    window["Yes"].update(visible = True, value = False)
    window["No"].update(visible = True, value = False)
    window["Confirm"].update(visible = True)
    window["no_bbox"].update(visible = False)
    enable_next_prev_buttons(window)

def hide_bbox_part(window):
    window["bbox_text"].update(visible = False)
    window["Yes"].update(visible = False)
    window["No"].update(visible = False)
    window["Confirm"].update(visible = False)

def enable_next_prev_buttons(window):
    window["Next"].update(visible = True)
    window["Prev"].update(visible = True)

def disable_next_prev_buttons(window):
    window["Next"].update(visible = False)
    window["Prev"].update(visible = False)

def show_annot_part(window):
    window["name_text"].update(visible = True)
    window["label"].update(visible = True, value = "")
    enable_next_prev_buttons(window)

def hide_annot_part(window):
    window["name_text"].update(visible = False)
    window["label"].update(visible = False)

def show_edit_part(window):
    window["edit_text"].update(visible = True)
    window["Edit"].update(visible = True)
    enable_next_prev_buttons(window)

def hide_edit_part(window):
    window["edit_text"].update(visible = False)
    window["Edit"].update(visible = False)

def show_edit_bbox_part(window):
    window["wrong_bbox_text"].update(visible = True)
    window["Edit"].update(visible = True)
    enable_next_prev_buttons(window)

def hide_edit_bbox_part(window):
    window["wrong_bbox_text"].update(visible = False)
    window["Edit"].update(visible = False)

def show_init(window):
    window["text1"].update(visible = True)
    window["image"].update(visible = True)
    window["file_list"].update(visible = True)
    hide_annot_part(window)
    show_bbox_part(window)
    hide_edit_bbox_part(window)
    hide_edit_part(window)
    enable_next_prev_buttons(window)
    window["folder"].update(visible = True)
    window["final_text"].update(visible = False)
    window["Quit"].update(visible = False)
    window["No Quit"].update(visible = False)
    window["folder_browser"].update(visible = True)
    window["Exit"].update(visible = True)

def show_exit(window):
    window["text1"].update(visible = False)
    window["image"].update(visible = False)
    window["file_list"].update(visible = False)
    hide_annot_part(window)
    hide_bbox_part(window)
    hide_edit_bbox_part(window)
    hide_edit_part(window)
    disable_next_prev_buttons(window)
    window["folder"].update(visible = False)
    window["final_text"].update("Thank you for annotating the images. The file with all the annotations is saved in " + os.getcwd() + "/annotated_data.csv")
    window["final_text"].update(visible = True)
    window["Quit"].update(visible = True)
    window["No Quit"].update(visible = True)
    window["folder_browser"].update(visible = False)
    window["Exit"].update(visible = False)



def mark_annotated(window, values, location, last_names, folder, name, label, df_annotated):
    if values["label"] != "":
        last_names[location] = "Already Annotated #" + str(location)
        window["file_list"].set_value([])
        window["file_list"].update(last_names)
        df_annotated = add_annotated_record(folder, name, label, df_annotated, window)
        values["label"] = ""
    return last_names, df_annotated

def mark_excluded(window, values, location, last_names, folder, name, df_excluded):
    last_names[location] = "Wrong Bounding Box #" + str(location) 
    window["file_list"].set_value([])
    window["file_list"].update(last_names)
    df_excluded = exclude_image(folder, name, df_excluded, window)
    return last_names, df_excluded

def check_if_annotate_data(df_annotated, fname, current_folder):
    names = df_annotated["Name_of_Image"].to_numpy()
    folders = df_annotated["Folder"].to_numpy()
    i = 0
    for i in range(len(names)):
        if folders[i] == current_folder:
            if names[i] == fname:
                return False

    return True


def handle_edition_bbox(df_excluded, name, folder):
    folders = df_excluded["Folder"].to_numpy()
    names = df_excluded["Name_of_Image"].to_numpy()

    for i in range(len(names)):
        if folders[i] == folder: 
            if names[i] == name:
                folders = np.delete(folders, i)
                names = np.delete(names, i)
                break

    d = {
        "Folder": folders, 
        "Name_of_Image":names
    }
    df_excluded = pd.DataFrame(d)
    display(df_excluded)
    return df_excluded

def handle_edition_annotated(df_annotated, name, folder):
    folders = df_annotated["Folder"].to_numpy()
    names = df_annotated["Name_of_Image"].to_numpy()
    labels = df_annotated["Label"].to_numpy()

    for i in range(len(folders)):
        if folders[i] == folder:
            if names[i] == name:
                folders = np.delete(folders, i)
                names = np.delete(names, i)
                labels = np.delete(labels, i)
                break

    d = {
        "Folder": folders, 
        "Name_of_Image":names,
        "Label":labels
    }

    df_annotated = pd.DataFrame(d)
    display(df_annotated)
    return df_annotated

def exclude_image(folder, name, df_excluded, window):
    if check_if_excluded_data(df_excluded, name, folder) == True:
        d_aux = {
            "Folder": [folder],
            "Name_of_Image":[name]
        }

        df_excluded_aux = pd.DataFrame(d_aux)
    
        df_excluded = df_excluded.append(df_excluded_aux)

        display(df_excluded)

    return df_excluded

def check_remove_exclusion(df_excluded, name, folder, window):
    removed = False
    if check_bbox_edition(df_excluded, name, folder, window) == True:
        df_excluded = handle_edition_bbox(df_excluded, name, folder)
        removed = True

    return removed, df_excluded

def check_remove_annotation(df_annotated, name, folder, window):
    removed = False
    if check_annotation_edition(df_annotated, name, folder, window) == True:
        df_annotated = handle_edition_annotated(df_annotated, name, folder)
        removed = True

    return removed, df_annotated


def check_if_excluded_data(df_excluded, fname, current_folder):
    names = df_excluded["Name_of_Image"].to_numpy()
    folders = df_excluded["Folder"].to_numpy()

    i = 0

    for i in range(len(names)):
        if folders[i] == current_folder:
            if names[i] == fname:
                return False

    return True


def search_location_in_listbox(name_to_locate, last_names):
    location = 0
    for i in range(len(last_names)):
        if last_names[i] == name_to_locate:
            location = i
    return location

def check_annotation_edition(df_annotated, orig_name, folder, window):
    if check_if_annotate_data(df_annotated, orig_name, folder) == True:
        hide_edit_part(window)
        hide_edit_bbox_part(window)
        hide_annot_part(window)
        show_bbox_part(window)
        return False
    else:
        hide_bbox_part(window)
        hide_annot_part(window)
        hide_edit_bbox_part(window)
        show_edit_part(window)
        return True

def check_bbox_edition(df_excluded, orig_name, folder, window):
    if check_if_excluded_data(df_excluded, orig_name, folder) == True:
        hide_edit_part(window)
        hide_edit_bbox_part(window)
        hide_annot_part(window)
        show_bbox_part(window)
        return False
    else:
        hide_bbox_part(window)
        hide_annot_part(window)
        hide_edit_part(window)
        show_edit_bbox_part(window)
        return True

def launch_UI():

    window = sg.Window("Image Viewer", layout)
    images = []
    location = 0
    
    folder = ""
    name = ""
    label = ""
    columns_annotated = ["Folder", "Name_of_Image", "Label"]
    columns_excluded = ["Folder", "Name_of_Image"]
    df_annotated = pd.DataFrame(columns = columns_annotated)
    df_excluded = pd.DataFrame(columns = columns_excluded)

    annotated = False

    last_names = []

    first_time = True

    # Execution Loop
    while True:
        event, values = window.read()

        if event == "Exit":

            hide_annot_part(window)
            hide_bbox_part(window)
            hide_edit_bbox_part(window)
            hide_edit_part(window)
            disable_next_prev_buttons(window)
            
            show_exit(window)
            

        if event == sg.WIN_CLOSED:
            break
        if event == "Quit":
            break
        if event == "No Quit":
            show_init(window)
            

        if event == "folder":
            folder = values["folder"]

            if first_time == False:
                save_annotated_csv(df_annotated)
                save_excluded_csv(df_excluded)
            else:
                first_time = False

            df_annotated = load_annotated_csv(folder)
            df_excluded = load_excluded_csv()
            images = parse_folder(folder)
            if images:
                load_image(images[0], window)
            
            fnames = [
                f 
                for f in images
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
            ]
            
            last_names = []
            last_names_orig = []
            pos = 0
            for last_name in fnames:
                aux = last_name.split("\\")
                if check_if_excluded_data(df_excluded, aux[1], folder) == True: 
                    if check_if_annotate_data(df_annotated, aux[1], folder) == True:
                        last_names.append(aux[1])
                        last_names_orig.append(aux[1])
                    else:
                        last_names.append("Already Annotated #" + str(pos))
                        last_names_orig.append(aux[1])

                else:
                    last_names.append("Wrong Bounding Box #" + str(pos))
                    last_names_orig.append(aux[1])
                pos += 1
                

            if images:
                window["file_list"].update(last_names)
                hide_annot_part(window)
                show_bbox_part(window)
                window["text1"].update(visible = True)

            if check_annotation_edition(df_annotated, last_names_orig[location], folder, window) == False:
                    check_bbox_edition(df_excluded, last_names_orig[location], folder, window)

        elif event == "file_list":

            if check_if_excluded_data(df_excluded, last_names_orig[location], folder) == True:
                if annotated == True:
                    last_names, df_annotated = mark_annotated(window, values, location, last_names, folder, last_names_orig[location], label, df_annotated)
                    annotated = False


            try:
                filename = os.path.join(
                    values["folder"], values["file_list"][0]
                )

                
                i = 0
                filename_prep = filename.split('\\')[1]

                hide_annot_part(window)
                hide_bbox_part(window)
                hide_edit_bbox_part(window)
                hide_edit_part(window)

                location = search_location_in_listbox(filename_prep, last_names)


                if check_annotation_edition(df_annotated, last_names_orig[location], folder, window) == False:
                    check_bbox_edition(df_excluded, last_names_orig[location], folder, window)
            
                load_image(images[location], window)

               

                
            except:
                pass

        if event == "Edit":
            hide_edit_part(window)
            hide_edit_bbox_part(window)
            hide_annot_part(window)
            show_bbox_part(window)

        if event == "Confirm":  
            if values["Yes"] == True:
                
                removed, df_excluded = check_remove_exclusion(df_excluded, last_names_orig[location], folder, window)
                if removed == True: 
                    window["file_list"].update(last_names)
                    hide_edit_bbox_part(window)
                    show_annot_part(window)
                    annotated = True
                else:
                    hide_bbox_part(window)
                    show_annot_part(window)
                    annotated = True

                

            if values["No"] == True:
                
                last_names, df_excluded = mark_excluded(window, values, location, last_names, folder, last_names_orig[location], df_excluded)
                removed, df_annotated = check_remove_annotation(df_annotated, last_names_orig[location], folder, window)

                if removed == True:
                    window["file_list"].update(last_names)
                    hide_edit_bbox_part(window)
                    hide_bbox_part(window)
                    window["no_bbox"].update(visible = True)
                    enable_next_prev_buttons(window)

                else:
                    hide_bbox_part(window)
                    window["no_bbox"].update(visible = True)
                    enable_next_prev_buttons(window)

                




        if event == "Next" and images:
            
            if check_if_excluded_data(df_excluded, last_names_orig[location], folder) == True:
                if annotated == True:
                    last_names, df_annotated = mark_annotated(window, values, location, last_names, folder, last_names_orig[location], label, df_annotated)
                    annotated = False
            
            if location == len(images) - 1:
                location = 0
            else:
                location += 1
            load_image(images[location], window)

            
           

            window["file_list"].update(set_to_index = location, scroll_to_index = location)
            

            hide_annot_part(window)
            show_bbox_part(window)

            if check_annotation_edition(df_annotated, last_names_orig[location], folder, window) == False:
                    check_bbox_edition(df_excluded, last_names_orig[location], folder, window)
            

        if event == "Prev" and images:
            if check_if_excluded_data(df_excluded, last_names_orig[location], folder) == True:
                if annotated == True:
                    last_names, df_annotated = mark_annotated(window, values, location, last_names, folder, last_names_orig[location], label, df_annotated)
                    annotated = False
            if location == 0:
                location = len(images) - 1
            else:
                location -= 1
            load_image(images[location], window)

            window["file_list"].update(set_to_index = location, scroll_to_index = location)

            hide_annot_part(window)
            show_bbox_part(window)

            if check_annotation_edition(df_annotated, last_names_orig[location], folder, window) == False:
                    check_bbox_edition(df_excluded, last_names_orig[location], folder, window)
        
        if len(last_names) > 0:
            name = last_names[location]
            label = values["label"]
        
    save_annotated_csv(df_annotated)
    save_excluded_csv(df_excluded)

    
    window.close()


if __name__ == "__main__":
    launch_UI()
