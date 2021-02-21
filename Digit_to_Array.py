# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:54:37 2021

@author: Akash MSI
"""


from PIL import ImageGrab, Image
import numpy as np
import cv2
import pytesseract    

thresh = 170

detected_tables_org = cv2.imread('detected_tables_org.jpg',0)
BW_detected_tables_org = cv2.threshold(detected_tables_org, thresh, 255, cv2.THRESH_BINARY)[1]

TS_detected_lines_horizontal = cv2.imread('TS_detected_lines_horizontal.jpg',0) 
BW_TS_detected_lines_horizontal = cv2.threshold(TS_detected_lines_horizontal, thresh, 255, cv2.THRESH_BINARY)[1]


TS_detected_lines_vertical = cv2.imread('TS_detected_lines_vertical.jpg',0) 
BW_TS_detected_lines_vertical = cv2.threshold(TS_detected_lines_vertical, thresh, 255, cv2.THRESH_BINARY)[1]


detected_table_from_folder = BW_detected_tables_org + BW_TS_detected_lines_horizontal +BW_TS_detected_lines_vertical

# cv2.imshow('detected_table_from_folder',detected_table_from_folder)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def text_it(arr):
    #config = ('-l eng --oem 1 --psm 6 ')
    # config = ('-l eng --oem 2 --psm 6 ')
    #config = ('-l eng --oem 2 --psm 6 ')
    config = r'--oem 3 --psm 6 outputbase digits'
    arr = np.uint8(arr)
    im_pil = Image.fromarray(arr)
    a,b = arr.shape
    text = ''
    if a and b:
        size =  b * 4 , a * 4
    
        im_resized = im_pil.resize(size, Image.ANTIALIAS)
        im_np = np.asarray(im_resized)
        im_bw = cv2.bitwise_not(im_np)
    
        kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(im_bw, -1, kernel_sharpening)
        text = pytesseract.image_to_string(im_pil , config = config)
        # print(text)
    return text

text_pop = np.zeros((9,9))


a,b = detected_table_from_folder.shape

for x in range (0, 9,1):
    for y in range (0, 9,1):
        sa =detected_table_from_folder[x*int(a/9):(x+1)*int(a/9) , y*int(b/9):(y+1)*int(b/9)]    
        text_out = text_it(np.array(sa))
        numeric_filter = filter(str.isdigit, text_out)
        numeric_string = "".join(numeric_filter)
        # text_1 = int(text)
        if numeric_string.isdigit():
            # print('Yes! Yes! Yes!' )
            if int(numeric_string) >0 and int(numeric_string) <10:
                text_pop[x,y] = int(numeric_string)
    
    
    
sudoku_que = text_pop.tolist()  

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 02:12:58 2021

@author: Akash MSI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:16:07 2021

@author: Akash MSI
"""

def display_board(grid_patt):   
    for i in range(1,10):
    
        for j in range(1,10):
            print(int(grid_patt[i-1][j-1]), end=" ")
        # print(f'j vale was: {j}')
            if j==9:
                print('\n' , end="")
            elif (j)%3 == 0:
                print('|', end=" ")
        if (i)%3 == 0 and i!=9:
            print('- - -   - - -   - - - ')     
            
            
def is_empty(i,j,grid_patt):
  if grid_patt[i-1][j-1] <= 0:
      return 1
  else: 
      return 0
 
    
def in_row(i,j,grid_patt,n):
    for k in range(1,10):
        if grid_patt[i-1][k-1] == n:
            return 1
    return 0  

        
def in_column(i,j,grid_patt,n):
    for k in range(1,10):
        if grid_patt[k-1][j-1] == n:
            return 1
    return 0  


def in_block(i,j,grid_patt,n):
    for k in range(int((i-1)/3)*3+1,int((i-1)/3)*3+4):
        for l in range(int((j-1)/3)*3+1,int((j-1)/3)*3+4):
            if grid_patt[k-1][l-1] == n:
                return 1
    return 0     
        
           
def empty_cell(grid_patt):
    temp=[]
    for i in range(1,10):
        for j in range(1,10):
            if grid_patt[i-1][j-1] <= 0:
                k = [i,j]
                # print(k)
                temp.append(k)
    # print(temp)            
    return temp    

 

def backtrack_solver(empty_cell_pos , grid_patt , i):
    
    # print(f'backtrack coutn: {i}')
    if i>= len(empty_cell_pos):
         print('Solved Board:')
         display_board(grid_patt)
         return grid_patt
    # else:
        
    a,b= empty_cell_pos[i]
    # print(f'a:{a} , b:{b}')
    for digit in range(1,10):
        if (in_row(a,b,grid_patt,digit) ==0 and in_column(a,b,grid_patt,digit)==0 and in_block(a,b,grid_patt,digit) ==0):
            # empty_list = empty_cell(grid_patt)
            # x,y = empty_list[0]
            # if x == a and y == b:    
            #     print('Going correct!!!!!')
            grid_patt[a-1][b-1] = digit
            # print('The intermediate solution is:')
            # display_board(grid_patt)
            #i=i+1
            # print(f'backtrack i: {i}')
            backtrack_solver(empty_cell_pos , grid_patt , i+1)
    # if digit ==9 and (in_row(a,b,grid_patt,9) ==1 or in_column(a,b,grid_patt,9)==1 or in_block(a,b,grid_patt,9) ==1):
    #print(f'a before zero:{a} , b before zero:{b}')
    grid_patt[a-1][b-1] = 0
                
    
    
# for i in range(0,empty_cnt):
#     if grid_patt[i-1][j-1] <= 0:

input_grid = sudoku_que
print('Orginal Board:')
display_board(input_grid)
    
empty_cell_pos = empty_cell(input_grid)
# empty_cnt = len(empty_cell_pos)
solved_grid = backtrack_solver(empty_cell_pos , input_grid , 0)


# print(empty_cell_pos)
# result = is_empty(1, 2, input_grid) 
# result_2 = in_row(1, 2, input_grid , 7) 
# result_3 = in_column(5, 7, input_grid ,4) 
# result_4 = in_block(5,7,input_grid,6)

# display_board(solved_grid)





        


    
    