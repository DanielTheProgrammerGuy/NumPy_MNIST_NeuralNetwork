import numpy as np
import pickle
import pygame as pg
import time
from NeuralNetwork import NeuralNetwork

bounding_box_scale = 1.7 #scale of bounding box around detected drawing

pg.init()
screen = pg.display.set_mode((400, 200))
font = pg.font.Font(None, 20)
screen.fill((0, 0, 0))

with open("neural_network.pkl", 'rb') as file:
    nn = pickle.load(file)

while True:
    screen = pg.display.set_mode((400, 200))
    running = True
    while not(pg.key.get_pressed()[pg.K_RETURN]) and running:
        for event in pg.event.get():
            # Check if the user clicked the window's close button
            if event.type == pg.QUIT:
                running = False

        if pg.mouse.get_pressed()[0]:
            if pg.key.get_pressed()[pg.K_SPACE]:
                colour = (0,0,0)
                pen_size = 20
            else:
                colour = (255,255,255)
                pen_size = 15
            pen_tip = pg.Rect(0,0,pen_size,pen_size)
            pen_tip.center = pg.mouse.get_pos()
            if connect:
                pg.draw.line(screen,colour,last_pos,pen_tip.center,pen_size)
            pg.draw.ellipse(screen, colour, pen_tip)
            connect = True
            last_pos = pg.mouse.get_pos()
        else:
            connect = False
        pg.display.flip()

    screen_array = np.mean(pg.surfarray.array3d(screen),axis=-1)/255
    number_of_digits = 0
    draw_array = []
    final_number = ""
    total_accuracy = 1
    while np.sum(np.where(screen_array > 0)) > 0:
        number_of_digits += 1
        digit_array = np.zeros_like(screen_array)
        points_in_digit = np.array([np.argwhere(screen_array > 0)[0]])
        while points_in_digit.size > 0:
            point = points_in_digit[0]
            digit_array[point[0], point[1]] = screen_array[point[0], point[1]]
            screen_array[point[0], point[1]] = 0
            points_in_digit = np.delete(points_in_digit, 0, axis=0)
            if point[1] != screen_array.shape[1]-1 and screen_array[point[0], point[1] + 1] > 0:
                points_in_digit = np.append(points_in_digit, [[point[0], point[1] + 1]], axis = 0)
            if point[1] != 0 and screen_array[point[0], point[1] - 1] > 0:
                points_in_digit = np.append(points_in_digit, [[point[0], point[1] - 1]], axis = 0)
            if point[1] != screen_array.shape[0]-1 and screen_array[point[0] + 1, point[1]] > 0:
                points_in_digit = np.append(points_in_digit, [[point[0] + 1, point[1]]], axis = 0)
            if point[0] != 0 and screen_array[point[0] - 1, point[1]] > 0:
                points_in_digit = np.append(points_in_digit, [[point[0] - 1, point[1]]], axis = 0)
            points_in_digit = np.unique(points_in_digit, axis = 0)
        xstart = np.where(np.sum(digit_array,axis = 1)>0)[0][0]
        ystart = np.where(np.sum(digit_array,axis = 0)>0)[0][0]
        xend = np.where(np.sum(digit_array,axis = 1)>0)[0][-1]+1
        yend = np.where(np.sum(digit_array,axis = 0)>0)[0][-1]+1

        xsize = xend - xstart
        ysize = yend - ystart

        square_size = max(xsize, ysize)
        if square_size*bounding_box_scale < 28:
            data_raw = np.zeros((28,28))
        else:
            data_raw = np.zeros((round(square_size*bounding_box_scale),round(square_size*bounding_box_scale)))
        data_raw[
            data_raw.shape[0]//2 - xsize//2 : data_raw.shape[0]//2 - xsize//2 + xsize,
            data_raw.shape[1]//2 - ysize//2 : data_raw.shape[1]//2 - ysize//2 + ysize
            ] = digit_array[xstart:xend,ystart:yend]

        data_processed = np.zeros((28,28))
        scaling = (data_raw.shape[0] // 28)
        for i in range(28):
            for j in range(28):
                data_processed[j,i] = np.mean(data_raw[(i)*scaling:(i+1)*scaling,(j)*scaling:(j+1)*scaling])
        data_processed = data_processed
        guess_output = nn.feedforward([data_processed.flatten()])
        guess = np.argmax(guess_output)

        total_accuracy *= 2*guess_output[0,guess]-np.sum(guess_output)
        if number_of_digits == 1:
            draw_array = data_processed
        else:
            draw_array = np.append(draw_array,data_processed,axis=1)
        final_number += str(guess)

    screen = pg.display.set_mode((280*number_of_digits, 280))
    for i in range(28*number_of_digits):
        for j in range(28):
            colour_value = int(255 * draw_array[j, i])
            pg.draw.rect(screen, (colour_value,colour_value,colour_value), pg.Rect(i * 10, j * 10, 10, 10))


    pg.draw.rect(screen,(0,255,0),(260*number_of_digits,0,20*number_of_digits,20),3)
    number = font.render(final_number, True, (0, 255, 0))
    number_rect = number.get_rect(center=(270*number_of_digits,10))
    screen.blit(number,number_rect)

    pg.draw.rect(screen, (0, 255, 0), (0, 0, 60, 20), 3)
    certainty = font.render(f"{total_accuracy*100:.2f}%", True, (0, 255, 0))
    certainty_rect = certainty.get_rect(center=(30,10))
    screen.blit(certainty,certainty_rect)

    runnning = True
    while running:
        for event in pg.event.get():
            # Check if the user clicked the window's close button
            if event.type == pg.QUIT or pg.key.get_pressed()[pg.K_ESCAPE]:
                running = False
        pg.display.update()
    

