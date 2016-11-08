# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 04:51:40 2016

@author: Kaibo Liu
"""

def getwidth(tree):
  if tree.left==None and tree.right==None: return 1
  return getwidth(tree.left)+getwidth(tree.right)

def getdepth(tree):
  if tree.left==None and tree.right==None: return 0
  return max(getdepth(tree.left),getdepth(tree.right))+1


from PIL import Image, ImageDraw, ImageFont

def drawtree(tree,jpeg='tree.jpg'):
    scale = 150
    w=getwidth(tree)*scale+50
    h=getdepth(tree)*scale+scale/2

    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    drawnode(draw,tree,w/2,20)
    img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
    #feature_name = ['(0)sepal length','(1)sepal width', '(2)petal length', '(3)petal width']
    feature_name = ['Feature 0','Feature 1', 'Feature 2', 'Feature 3']
    class_name   = ['class 0','class 1','class 2']
    fnt = ImageFont.truetype("./arial.ttf",22)
    scale = 150
    
    if tree.Class==None:
        # Get the width of each branch
        w1=getwidth(tree.left)*scale
        w2=getwidth(tree.right)*scale

        # Determine the total space required by this node
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        # Draw the condition string
        draw.text((x-50,y-20),feature_name[tree.feature]+' < '+str(tree.threshold)+' ?',font=fnt,fill=(0,0,0))

        # Draw links to the branches
        draw.line((x,y,left+w1/2,y+scale),fill=(255,0,0),width=3)
        draw.line((x,y,right-w2/2,y+scale),fill=(95, 57, 191),width=3)
    
        # Draw the branch nodes
        drawnode(draw,tree.left,left+w1/2,y+scale)
        drawnode(draw,tree.right,right-w2/2,y+scale)
    else:
        txt = class_name[int(tree.Class[0])]+':'+str(tree.Class[1])
        if len(tree.Class) > 2:
            txt += ' of '+str(tree.Class[2])
        #txt=' \n'.join(['%s:%d'%v for v in tree.Class.items()])
        draw.text((x-50,y),txt,font=fnt,fill=(0,0,0))