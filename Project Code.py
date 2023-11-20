from tkinter import *
import tkinter.messagebox
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename,asksaveasfilename
import os
import subprocess
from tkinter import filedialog
import PIL
from PIL import Image
import cv2
import numpy as np
import copy
import shutil
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from os import path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math

root = Tk()
root.title("Text Recognization Using Machine Learning")
root.geometry("1000x800")

def check_op(): #classifier() function called in this function 
    proc_status.destroy()
    if dir_sel == 1:
        classifier(cust_des)
    if dir_sel == 0:
        classifier(def_des)
    op_fold = Button(text="Open Output Folder", command=op_folder)
    op_fold.pack(pady=10)

def processing(): #croop(), b_w(), maain(), sepchar() functions called in this function 
    global img_frame,proc_status,proc_img
    if dir_sel == 1:
         croop(img_name)
         b_w(img_name)
         getchar_output = maain(cust_des, img_name)
         sepchar(getchar_output)
    if dir_sel == 0:
         croop(img_name)
         b_w(img_name)
         getchar_output = maain(def_des, img_name)
         sepchar(getchar_output)
    proc_status = Label(text="Processing...",relief=SUNKEN,anchor=W)
    proc_status.pack(side=BOTTOM,fill="x",pady=10)
    check_op()

def cont_dev():
    dev_info = tkinter.messagebox.showinfo("Contact Developer","For any Query or Feedback Contact Developer at :\ngroup9_becse@xyz.com")

def about():
    disin = tkinter.messagebox.showinfo("About",
                                        "This Software is Develop by BE-CSE (2020-21) Project Group 9 \n"
                                        "Topic : Text Recognization Using Machine Learning")
                                         
def work_space():
    global cust_des,def_des,dir_sel,add_space
    response_work_space = tkinter.messagebox.askyesno("Create Project Workspace","Would You Like to Create Custom Project Workspace ?\nOr Use Default Workspace C//Project ?\n"
                                                                                 "\n[Press Yes for Custom Workspace OR No for Default Workspace]")
    if response_work_space:
        work_space_path = filedialog.askdirectory()
        add_space = work_space_path + "/Demo_Project_1"   
        src = "C://Cpy_demo"  #Path of SmallAZVersion0 Folder Here
        cust_des = add_space
        os.makedirs(add_space)
        shutil.rmtree(cust_des)
        shutil.copytree(src,cust_des)
        tkinter.messagebox.showinfo("Create Project Workspace","Custom Workspace Created Successfully !")
        dir_sel = 1
    else:
        src = "C://Cpy_demo"   #Path of SmallAZVersion0 Folder Here 
        def_des = "C://New//Project"
        shutil.copytree(src,def_des)
        tkinter.messagebox.showinfo("Create Project Workspace","Defualt Workspce Used For Project !")
        dir_sel = 0


def new_proj():  
    global file,photo,img_frame,img_ext,proc_img,img_name
    work_space()
    file = askopenfilename(defaultextension=".png",filetypes=[("Image","*.png"),("Image","*.jpg")])
    img_name = os.path.abspath(file)
    grb,img_ext = os.path.splitext(file)
    
    if img_ext ==".png":
        jpg_image = Image.open(img_name)
        if dir_sel:
            backup_img = jpg_image.copy()
            backup_img_path = add_space + "backup_img.png"
            backup_img.save(backup_img_path,"PNG")
        else:
            backup_img = jpg_image.copy()
            backup_img_path = def_des + "backup_img.png"
            backup_img.save(backup_img_path,"PNG")
        jpg_image = jpg_image.resize((500, 500), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(jpg_image)
        label = Label(image=photo)
        label.pack(pady=20)
        img_frame = Frame(root)
        img_frame.pack()
        proc_img = Button(img_frame,text="Process Image",command=processing)
        proc_img.pack()

    elif img_ext ==".jpg":
        jpg_image = Image.open(img_name)
        if dir_sel:
            backup_img = jpg_image.copy()
            backup_img_path = add_space + "backup_img.jpg"
            backup_img.save(backup_img_path,"JPEG")
        else:
            backup_img = jpg_image.copy()
            backup_img_path = def_des + "backup_img.jpg"
            backup_img.save(backup_img_path,"JPEG")
        jpg_image = jpg_image.resize((500, 500), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(jpg_image)
        label = Label(image=photo)
        label.pack(pady=20)
        img_frame = Frame(root)
        img_frame.pack()
        proc_img = Button(img_frame,text="Process Image",command=processing)
        proc_img.pack()

    else:
        not_supt = tkinter.messagebox.showinfo("File Type Error","File You Tried Opening in not Supported")


def op_folder():
    subprocess.Popen('explorer "C:\Proj\Output"')

def close():
    answer = tkinter.messagebox.askokcancel("Confirm Exit","Are You Sure ?\nUnfinished Processes Data will be Lost")
    if(answer):
        quit()


def croop(Image_path):
    try:
        import cv2,copy,tkinter
        points = []
        flag = 0
        img = cv2.imread(Image_path, 1)
        root = tkinter.Tk()
        Image_hight,Image_width = img.shape[:2]
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        screen_height = screen_height*0.9
        resize_w, resize_h = Image_width,Image_hight
        while resize_h > screen_height or resize_w > screen_width:
            resize_h -=10
            resize_w -=10

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x,y])
                print(x,y)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.resizeWindow('image', resize_w, resize_h)
        cv2.setMouseCallback('image', click_event)
        if cv2.waitKey(0) & 0xFF == 13:
            flag = 1
            cv2.destroyAllWindows()
        
        if flag == 0:
            return
        
        x1,y1 = points[-2][0],points[-2][1]
        x2,y2 = points[-1][0],points[-1][1]
        img = img[y1:y2, x1:x2]
        cv2.imwrite(Image_path,img)

        img = cv2.imread(Image_path, 1)
        resize_h, resize_w = img.shape[:2]
        while resize_h > screen_height or resize_w > screen_width:
            resize_h -=10
            resize_w -=10

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.resizeWindow('image', resize_w, resize_h)
        if cv2.waitKey(0) & 0xFF == 13:
            cv2.destroyAllWindows()
    except:
        pass

def b_w(path):
    try:
        image=cv2.imread(path)

        resolution=image.shape
        original_resx=resolution[0]
        original_resy=resolution[1]
        average=image.mean(axis=0).mean(axis=0)

        print(original_resx)
                
        x=0
        y=0
        k=0
        while(1):
            if(k==0):
                start_of_y=y
                k=k+1
            if(x>=original_resx+2) and (y<original_resy+242): #<=
                x=0
                y=start_of_y+530
                k=0
            if(x>=original_resx and y>=original_resy): ##need to edit this static to based on image size Previous code if(x>=3638 and y>=2900):
                break
                
            else:
                sliced_image=image[x:x+242,y:y+530]
                resolution=sliced_image.shape
                resx=resolution[0]
                resy=resolution[1]
                average=sliced_image.mean(axis=0).mean(axis=0)
                
                deltaB=numpy.nan_to_num((average[0])-148)
                deltaG=numpy.nan_to_num((average[1])-138)
                deltaR=numpy.nan_to_num((average[2])-142)
                xB=(40*deltaB)/20
                xG=(22*deltaG)/34
                xR=(50*deltaR)/30
                alphaB=math.ceil(100+xB)
                alphaG=math.ceil(100+xG)
                alphaR=math.ceil(100+xR)
                
                print("old alphas")
                print("alphaB="+str(alphaB)+"  alphaG="+str(alphaG)+"  alphaR="+str(alphaR))
                
                if(alphaB>0 and alphaG>0 and alphaR>0) :	
                    #loading vector
                    vector=[0,0,0]
                    #Loading values in array
                    vals=[alphaB,alphaG,alphaR]
                    vals.sort(reverse=True)
                    Largest=vals[0]
                    #finding smallest among alpha(BGR)
                    if(alphaB<alphaG) and (alphaB<alphaR):
                        pump=Largest-alphaB
                        alphaB=Largest
                        vector[0]=1
                        #compare G & R
                        if(alphaG>alphaR) :# means G is greatest among all
                            if(alphaG<100):
                                alphaG=alphaG+numpy.nan_to_num(pump/3)
                            else:
                                alphaG=alphaG+numpy.nan_to_num(pump/6)
                            vector[1]=1
                        else:
                            if(alphaR<100):
                                alphaR=alphaR+numpy.nan_to_num(pump/3)
                            else:
                                alphaR=alphaR+numpy.nan_to_num(pump/6)		
                            vector[2]=1
                    elif(alphaG<alphaB) and (alphaG<alphaR):
                        pump=Largest-alphaG
                        alphaG=Largest
                        vector[1]=1
                        #compare B & R
                        if(alphaB>alphaR): #means B is greatest of all
                            if(alphaB<100):
                                alphaB=alphaB+numpy.nan_to_num(pump/3)
                            else:
                                alphaB=alphaB+numpy.nan_to_num(pump/6)
                            vector[0]=1	
                        else:
                            if(alphaR<100):
                                alphaR=alphaR+numpy.nan_to_num(pump/3)
                            else:		
                                alphaR=alphaR+numpy.nan_to_num(pump/6)
                            vector[2]=1
                    else:
                        pump=Largest-alphaR
                        alphaR=Largest
                        vector[2]=1
                        #compare B & G
                        if(alphaB>alphaG): #means B is greatest of all
                            if(alphaB<100):
                                alphaB=alphaB+numpy.nan_to_num(pump/3)
                            else:
                                alphaB=alphaB+numpy.nan_to_num(pump/6)
                            vector[0]=1
                        else:
                            if(alphaG<100):
                                alphaG=alphaG+numpy.nan_to_num(pump/3)
                            else:
                                alphaG=alphaG+numpy.nan_to_num(pump/6)
                            vector[1]=1
                    #now computing the middle value
                    if(vector[0]==0):
                        alphaB=numpy.nan_to_num((alphaG+alphaR)/2)
                    elif(vector[1]==0):
                        alphaG=numpy.nan_to_num((alphaB+alphaR)/2)
                    else:
                        alphaR=numpy.nan_to_num((alphaG+alphaB)/2)
                    print("new alphas")
                    print("alphaB="+str(alphaB)+"  alphaG="+str(alphaG)+"  alphaR="+str(alphaR))
                        
                else:	#now load the code from sliced_avg
                    vector=[0,0,0]
                    #Loading values in array
                    vals=[alphaB,alphaG,alphaR]
                    vals.sort(reverse=True)
                    Largest=vals[0]
                    #finding smallest among alpha(BGR)
                    if(alphaB<alphaG) and (alphaB<alphaR):
                        #equalte lowest value to the 1/3rd of largest;here B is lowest
                        alphaB=numpy.nan_to_num(Largest/3)
                        vector[0]=1
                        #compare G & R
                        if(alphaG>alphaR) :# means G is greatest among all
                            #alphaG=alphaG+numpy.nan_to_num(pump/3)
                            vector[1]=1
                        else:
                            #alphaR=alphaR+numpy.nan_to_num(pump/3)		
                            vector[2]=1
                    elif(alphaG<alphaB) and (alphaG<alphaR):
                        alphaG=numpy.nan_to_num(Largest/3)
                        vector[1]=1
                        #compare B & R
                        if(alphaB>alphaR): #means B is greatest of all
                            #alphaB=alphaB+numpy.nan_to_num(pump/3)
                            vector[0]=1	
                        else:
                            #alphaR=alphaR+numpy.nan_to_num(pump/3)		
                            vector[2]=1
                    else:
                        alphaB=numpy.nan_to_num(Largest/3)
                        vector[2]=1
                        #compare B & G
                        if(alphaB>alphaG): #means B is greatest of all
                            #alphaB=alphaB+numpy.nan_to_num(pump/3)
                            vector[0]=1
                        else:
                            #alphaG=alphaG+numpy.nan_to_num(pump/3)
                            vector[1]=1
                    #now computing the middle value
                    if(vector[0]==0):
                        alphaB=numpy.nan_to_num((alphaG+alphaR)/2)
                    elif(vector[1]==0):
                        alphaG=numpy.nan_to_num((alphaB+alphaR)/2)
                    else:
                        alphaR=numpy.nan_to_num((alphaG+alphaB)/2)
                        
                    alphaB=math.ceil(alphaB)
                    alphaG=math.ceil(alphaG)
                    alphaR=math.ceil(alphaR)
                    print("new alphas")
                    print("alphaB="+str(alphaB)+"  alphaG="+str(alphaG)+"  alphaR="+str(alphaR))
                    
                #	
                i=0
                for i in range(0,resx):
                    for j in range(0,resy):
                        extract_color=sliced_image[i][j]
                        R=extract_color[2]
                        G=extract_color[1]
                        B=extract_color[0]
                        if(B<=alphaB and G<=alphaG and R<=alphaR) :#then Set pixel to black  
                            sliced_image[i][j]=(0,0,0)
                        else:
                            sliced_image[i][j]=(255,255,255)
                
                
                
                #Now sliced image is ready with B/W image
                image[x:x+242,y:y+530]=sliced_image	
                x=x+242
                #saving image
                cv2.imwrite(r"E:\Scion_of_Ikshavaku_tests\res_bright1.jpg",image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass        

def maain(working_di, path, space_width=-1):

    def getimage(path=None):
        if path == None:
            return 'Image not Found'
        image1 = cv2.imread(path)
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            199,
            5,
            )
        cv2.imwrite('temp/temp/' + path[:-4] + '-temp' + path[-4:], img)
        img = Image.open('temp/temp/' + path[:-4] + '-temp'
                         + path[-4:], 'r')
        return img

    def get_image_from_coordinates(coordinates_of_char, im, padding=0):
        list_char = []
        (w, h) = im.size
        for i in range(len(coordinates_of_char)):
            (start, end) = coordinates_of_char[i]
            im1 = im.crop((start, 0, end, h))
            if isblank(im1):
                continue
            list_char.append(im1)
        return list_char

    def cut_cornors_of_char(list_char):
        listt = []
        for i in range(len(list_char)):
            (a, b) = find_cornors(list_char[i])
            if a >= 120:
                continue
            listt.append(list_char[i].crop((0, a, list_char[i].size[0],
                         b)))
        return listt

    def find_cornors(im):
        l = im.load()
        (w, h) = im.size
        upper = 0
        lower = h
        flag = 0
        for i in range(h):
            for j in range(w):
                if l[j, i] == 255:
                    upper = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                if l[j, i] == 255:
                    lower = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        return (upper, lower + 1)

    def resize(image, window_height=128 - 20):
        try:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            window_width = window_height / aspect_ratio
            image = cv2.resize(image, (int(window_height),
                               int(window_width)))
            return image
        except:
            return image

    def make128(im):
        im = resize(np.array(im))
        while im.shape[0] > 110:
            im = resize(im, im.shape[1] - 3)
        im = Image.fromarray(np.uint8(im))
        (w, h) = im.size
        left = (128 - w) // 2
        top = (128 - h) // 2
        result = Image.new(im.mode, (128, 128), 255)
        result.paste(im, (left, top))
        return result

    def sharpen_image(img_bw):
        img_bw.save('temp.png')
        img = cv2.imread('temp.png')
        img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        > 150).astype('uint8')
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        mask = np.dstack([mask, mask, mask]) / 255
        out = img * mask
        (thresh, blackAndWhiteImage) = cv2.threshold(out, 127, 255,
                cv2.THRESH_BINARY)
        return Image.fromarray(np.uint8(blackAndWhiteImage))

    def find_blank_lines_in_word(im):
        list_of_lines = []
        l = im.load()
        (w, h) = im.size
        for i in range(w):
            blank = 1
            for j in range(h):
                if l[i, j] == 0:
                    blank = 0
                    break
            if blank == 1:
                list_of_lines.append(1)
            else:
                list_of_lines.append(0)
        list_of_lines.append(1)
        return list_of_lines

    def find_blank_lines(im):
        list_of_lines = []
        l = im.load()
        (w, h) = im.size
        for j in range(h):
            blank = 1
            for i in range(w):
                if l[i, j] != 255:
                    blank = 0
            if blank == 1:
                list_of_lines.append(1)
            else:
                list_of_lines.append(0)
        return list_of_lines

    def generate_coordinates(list_of_lines):
        coordinates_of_lines = []
        temp = list_of_lines[0]
        count = 0
        for i in range(1, len(list_of_lines)):
            if list_of_lines[i] == temp:
                count += 1
            else:
                coordinates_of_lines.append([i - 1 - count, i - 1])
                count = 0
                temp = 1 - temp
        return coordinates_of_lines

    def isblank(img, percent=2):
        l2 = np.array(img)
        q = l2.shape
        if q == ():
            return 1
        a_percent = int(np.sum(l2 == 0) / (l2.shape[0] * l2.shape[1])
                        * 100)
        if a_percent <= percent:
            return 1
        return 0

    def save_image_lines(
        coordinates_of_lines,
        im,
        padding=0,
        percent_of_blank=0,
        ):
        list_of_lines = []
        count = 0
        (w, h) = im.size
        for i in range(len(coordinates_of_lines)):
            (start, end) = coordinates_of_lines[i]
            y1 = max(start - padding, 0)
            y2 = min(end + padding, h)
            im1 = im.crop((0, y1, w, y2))
            if isblank(im1, percent_of_blank):
                continue
            list_of_lines.append(im1)
        return list_of_lines

    def find_blank_lines2(im):
        list_of_words = []
        (w, h) = (im.shape[1], im.shape[0])
        for i in range(w):
            blank = 1
            for j in range(h):
                if im[j, i] != 255:
                    blank = 0
            if blank == 1:
                list_of_words.append(1)
            else:
                list_of_words.append(0)
        return list_of_words

    def generate_coordinates_of_words(list_of_words, space_width=4):
        lenn = len(list_of_words)
        i = 0
        dic = {}
        while i < lenn:
            count = 0
            start = i
            if i < lenn and list_of_words[i] == 1:
                while i < lenn and list_of_words[i] == 1:
                    i += 1
                    count += 1
                dic[start] = count
            else:
                i += 1
        list_of_coor = list(dic.items())
        i = 0
        l = len(list_of_coor)
        while i < len(list_of_coor):
            if list_of_coor[i][1] > space_width:
                i += 1
            else:
                del list_of_coor[i]
        temp = []
        for i in range(1, len(list_of_coor)):
            temp.append([list_of_coor[i - 1][1] + list_of_coor[i
                        - 1][0], list_of_coor[i][0]])
        return temp

    def save_image_words(coordinaes_of_words, im, padding=0):
        list_of_words = []
        count = 0
        (w, h) = (im.shape[1], im.shape[0])
        for i in range(len(coordinaes_of_words)):
            (start, end) = coordinaes_of_words[i]
            x1 = max(start - padding, 0)
            x2 = min(end + padding, w)
            im1 = im[:, x1:x2]
            list_of_words.append(im1)
        return list_of_words

    def get_list_of_words(each_line, space_width):
        list_of_blank_lines = find_blank_lines2(np.array(each_line))
        list_of_coordinaes_of_words = \
            generate_coordinates_of_words(list_of_blank_lines,
                space_width)
        return save_image_words(list_of_coordinaes_of_words,
                                np.array(each_line))

    def get_word_from_lines(list_lines, space_width):
        list_of_words = []
        for each_line in list_lines:
            a = get_list_of_words(each_line, space_width)
            list_of_words.append(a)
        return list_of_words

    def add_margin1(im, top):
        (w, h) = im.size
        (right, bottom, left) = (top, top, top)
        new_width = w + right + left
        new_height = h + top + bottom
        result = Image.new(im.mode, (new_width, new_height), 255)
        result.paste(im, (left, top))
        return result

    def cut_cornors_of_char_all_side(list_char):
        listt = []
        for i in range(len(list_char)):
            (l, r, t, b) = find_cornors_all_side(list_char[i])
            if t >= 120:
                continue
            listt.append(list_char[i].crop((l, t, r, b)))
        return listt

    def find_cornors_all_side(im):
        l = im.load()
        (w, h) = im.size
        left = 0
        upper = 0
        lower = h
        right = h
        flag = 0
        for i in range(h):
            for j in range(w):
                if l[j, i] == (255, 255, 255):
                    upper = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                if l[j, i] == (255, 255, 255):
                    lower = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(w - 1, -1, -1):
            for j in range(h - 1, -1, -1):
                if l[i, j] == (255, 255, 255):
                    right = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(w):
            for j in range(h):
                if l[i, j] == (255, 255, 255):
                    left = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break
        return (left, right, upper, lower)

    def find_space_width(line):
        space_width = find_blank_lines2(np.array(line))
        space = []
        i = 0
        while i < len(space_width):
            if space_width[i] == 1:
                c = 0
                while i < len(space_width) and space_width[i] == 1:
                    c += 1
                    i += 1
                space.append(c)
            else:
                i += 1
        del space[0]
        del space[-1]
        return min(space) + 4

    ttemp = working_di
    os.chdir(working_di)
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (thresh, im) = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
    im = Image.fromarray(np.uint8(im))
    list_of_blank_lines = find_blank_lines(im)
    coordinates_of_lines = generate_coordinates(list_of_blank_lines)
    list_lines = save_image_lines(coordinates_of_lines, im, 0)
    if space_width == -1:
        space_width = find_space_width(list_lines[0])
    list_of_words = get_word_from_lines(list_lines, space_width)
    if 'temp' in os.listdir():
        shutil.rmtree('temp')
    os.mkdir('temp')
    os.chdir(working_di + r'/temp')
    count = 0
    for i in range(len(list_of_words)):
        for j in range(len(list_of_words[i])):
            name = str(count) + r'.png'
            im = Image.fromarray(np.uint8(list_of_words[i][j]))
            im = add_margin1(im, 3)
            im.save(name)
            count += 1
    os.chdir(working_di)
    working_di = working_di + r'\temp'
    temp_di = working_di + r'\temp'
    word_di = temp_di
    os.chdir(working_di)
    if 'temp' in os.listdir():
        shutil.rmtree(temp_di)
    link = os.listdir()
    os.makedirs(temp_di + r'/temp')
    count = 0
    for i in range(len(link)):
        os.chdir(working_di)
        path = link[i]
        im = getimage(path)
        blank_lines = find_blank_lines_in_word(im)
        coordinates = generate_coordinates(blank_lines)
        list_char = get_image_from_coordinates(coordinates, im)
        list_char = cut_cornors_of_char(list_char)
        processed_chars = []
        for i in range(len(list_char)):
            im = make128(list_char[i])
            processed_chars.append(sharpen_image(im))
        os.chdir(word_di)
        name_of_word = path[:-4]
        os.mkdir(name_of_word)
        for i in range(len(processed_chars)):
            processed_chars[i] = \
                sharpen_image(Image.fromarray(cv2.blur(np.array(processed_chars[i]),
                              (10, 10))))
            processed_chars[i].save(name_of_word + '/' + str(i)
                                    + path[-4:])
    os.chdir(ttemp)
    shutil.rmtree(word_di + r'\temp')
    original = ttemp + r"\temp\temp"
    target = ttemp + '\\words'
    if 'words' in os.listdir(ttemp):
        shutil.rmtree(ttemp + '\\words')
    shutil.move(original, target)
    shutil.rmtree(ttemp + r'\temp')
    os.chdir(ttemp + '\\words')
    a = list(os.listdir())
    if 'temp.png' in a:
        a.remove('temp.png')
        os.remove('temp.png')
    a = list(map(int, a))
    a.sort()
    a = list(map(str, a))
    a = list(map(lambda x: [x], a))
    with open('words_dictionary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wordfile'])
        writer.writerows(a)
    os.chdir(ttemp)
    return target


# maain(r"C:\Users\shree\Downloads\t",r"C:\Users\shree\Downloads\t\2.png")

def sepchar(working_di):
    
    def cut_cornors_of_char_all_side(list_char):
        listt = []
        for i in range(len(list_char)):
            (l, r, t, b) = find_cornors_all_side(list_char[i])
            if t >= 120:
                continue
            listt.append(list_char[i].crop((l, t, r, b)))
        return listt

    def find_cornors_all_side(im):
    
        l = im.load()
        (w, h) = im.size
        left = 0
        upper = 0
        lower = h
        right = h
        flag = 0

        for i in range(h):
            for j in range(w):
                if l[j, i] == (255, 255, 255):
                    upper = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break

        flag = 0
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                if l[j, i] == (255, 255, 255):
                    lower = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break

        flag = 0
        for i in range(w - 1, -1, -1):
            for j in range(h - 1, -1, -1):
                if l[i, j] == (255, 255, 255):
                    right = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break

        flag = 0
        for i in range(w):
            for j in range(h):
                if l[i, j] == (255, 255, 255):
                    left = i
                else:
                    flag = 1
                    break
            if flag == 1:
                break

        return (left, right, upper, lower)

    def is_not_i(im):
        try:
            im = cut_cornors_of_char_all_side([im])
            if im[0].size[1] / im[0].size[0] < 2:
                return 1
            return 0
        except:
            return 0

    def seprate_char(im):
        images = []
        q_compute = []
        for i in range(len(im)):
            if is_not_i(im[i]):
                try:

                    def floodfill(
                        i,
                        j,
                        new_color,
                        old_color,
                        ):
                        if l1[i, j] == old_color:
                            l1[i, j] = new_color
                            l2[i, j] = old_color
                            q_compute.append([i + 1, j])
                            q_compute.append([i - 1, j])
                            q_compute.append([i, j + 1])
                            q_compute.append([i, j - 1])

                            floodfill(i + 1, j, new_color, old_color)
                            floodfill(i - 1, j, new_color, old_color)
                            floodfill(i, j + 1, new_color, old_color)
                            floodfill(i, j - 1, new_color, old_color)

                    flag = 0
                    first = copy.deepcopy(im[i])
                    second = Image.new('RGB', (128, 128), 'white')
                    l1 = first.load()
                    l2 = second.load()

                    for ii in range(first.size[0]):
                        for j in range(first.size[1]):
                            if l1[ii, j] == (0, 0, 0):
                                flag = 1
                                floodfill(ii, j, (255, 255, 255), (0,
                                        0, 0))
                                images.append(copy.deepcopy(second))
                                images.append(copy.deepcopy(first))
                                break
                        if flag == 1:
                            break
                except:
                    try:
                        while q_compute != []:
                            (temp_i, temp_j) = q_compute.pop()
                            floodfill(temp_i, temp_j, (255, 255, 255),
                                    (0, 0, 0))
                        images.append(copy.deepcopy(second))
                        images.append(copy.deepcopy(first))
                    except:

                        images.append(im[i])
            else:
                images.append(im[i])
        return images

    def make128_rgb(im):
    
        try:
            im = resize(np.array(im))
            count_resize = 1
            while im.shape[0] > 110 and count_resize < 40:
                im = resize(im, im.shape[1] - 3)
                count_resize += 1

            im = Image.fromarray(np.uint8(im))
            (w, h) = im.size
            left = (128 - w) // 2
            top = (128 - h) // 2
            result = Image.new(im.mode, (128, 128), (255, 255, 255))
            result.paste(im, (left, top))
            return result
        except:
            return im

    def sharpen_image(img_bw):
    
        img_bw.save('temp.png')
        img = cv2.imread('temp.png')
        img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        > 150).astype('uint8')

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        mask = np.dstack([mask, mask, mask]) / 255
        out = img * mask

        (thresh, blackAndWhiteImage) = cv2.threshold(out, 127, 255,
                cv2.THRESH_BINARY)
        return Image.fromarray(np.uint8(blackAndWhiteImage))

    def resize(image, window_height=128 - 20):
    
        try:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            window_width = window_height / aspect_ratio
            image = cv2.resize(image, (int(window_height),
                               int(window_width)))
            return image
        except:
            return image


    os.chdir(working_di)
    names = os.listdir()
    names.remove('words_dictionary.csv')

    for i in range(len(names)):
        try:
            os.chdir(names[i])
            image_names = os.listdir()
            collect = []

            for j in image_names:
                images = Image.open(j)
                seprated_images = seprate_char([images])
                for img in seprated_images:
                    collect.append(img)

            for img22 in range(len(collect)):
                collect[img22] = cut_cornors_of_char_all_side([collect[img22]])
                collect[img22] = collect[img22][0]
                collect[img22] = make128_rgb(collect[img22])
                collect[img22] = sharpen_image(Image.fromarray(cv2.blur(np.array(collect[img22]),(5, 5))))
                collect[img22].save(str(img22) + '.png')

            if 'temp.png' in os.listdir():
                os.remove('temp.png')
            os.chdir(working_di)

        except:
            os.chdir(working_di)

    os.chdir(working_di)
    folders = os.listdir()
    folders.remove("words_dictionary.csv")
    for folder in folders:
        os.chdir(folder)
        if 'temp.png' in os.listdir():
            os.remove('temp.png')
        os.chdir(working_di)


def classifier(working_di, threshold = 5.0):	
	model_path =  working_di + '\\' + 'SmallAZVersion2'
	dictionary_path =  working_di + '\\' + 'words' + '\\' + 'words_dictionary.csv'
	word_directory = working_di + '\\' +'words' + '\\'
	output_path = working_di + '\\' 
	
	'''import os
	import tensorflow as tf
	from tensorflow import keras
	# from tensorflow.keras import layers, regularizers
	# from PIL import Image
	from os import path
	# import matplotlib.pyplot as plt
	# import PIL
	import numpy as np
	# import matplotlib.image as mpimg
	import cv2
	import pandas as pd'''

	word_dict = {
		1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i', 10:'j', 11:'k',
		12:'l', 13:'m', 14:'n', 15:'o', 16:'p', 17:'q', 18:'r', 19:'s',20:'t', 21:'u',
		22:'v', 23:'w', 24:'x', 25:'y', 26:'z'
				}
			
	df = pd.read_csv(dictionary_path)
	wordtest_index = df['wordfile']
	
	pred_128model = keras.models.load_model(model_path)
	
	#-----------------
	text = ''
	for i in range(len(wordtest_index)):
		letter_index = 0
		while(True):
			letter_directory = word_directory + str(wordtest_index[i]) + '\\' + str(letter_index) + '.png'
		
			if(path.exists(letter_directory)):
				letter_img = cv2.imread(letter_directory, 0)
				img_array = tf.keras.preprocessing.image.img_to_array(letter_img)
				img_batch = np.expand_dims(img_array, axis=0)
				prediction = pred_128model.predict(img_batch)
      
				#print(prediction)
				x = list(prediction[0])
				word_index = x.index(max(x)) + 1
				#print("index:",word_index)
				#print("word:",word_dict[word_index])
      
				if max(x) >= threshold:
					text = text + word_dict[word_index]
				
				letter_index = letter_index + 1
    
			else:
				text =  text + ' '
				break
				
	new_text_file_path = output_path + 'word_to_text' + '.txt'
	word_to_text = open(new_text_file_path, 'w')
	word_to_text.write(text)
	word_to_text.close()

mainmenu = Menu(root)

filemenu = Menu(mainmenu,tearoff=0)
filemenu.add_command(label="New Project",command=new_proj)
filemenu.add_command(label="Open Output Folder",command=op_folder)
filemenu.add_separator()
filemenu.add_command(label="Exit",command=close)
helpmenu = Menu(mainmenu,tearoff=0)
helpmenu.add_command(label="Contact Developer",command=cont_dev)
helpmenu.add_separator()
helpmenu.add_command(label="About",command=about)
mainmenu.add_cascade(label="File",menu=filemenu)
mainmenu.add_cascade(label="Help",menu=helpmenu)
root.config(menu=mainmenu)


f1 = Frame(root,bg="grey",borderwidth=6,relief=RIDGE)
f1.pack(side=LEFT,fill="y")

f2 = Frame(root,bg="grey",borderwidth=6,relief=RIDGE)
f2.pack(side=BOTTOM,fill="x")

status = Label(f2,text="*Text Recognization Using Machine Learning\nVersion v0.0",fg="blue",font="Helvetica 10 bold")
status.pack(fill=X)

b1 = Button(f1,text="New Project",fg="black",command=new_proj,relief=RAISED)
b1.pack(pady=20,padx=20)

b2 = Button(f1,text="Open Output \nFolder",fg="black",command=op_folder,relief=RAISED)
b2.pack(pady=15,padx=20)

b3 = Button(f1,text="Exit",fg="red",command=close,relief=RAISED)
b3.pack(pady=15,padx=20)

root.mainloop()