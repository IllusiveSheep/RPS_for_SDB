import random

import PySimpleGUI as sg
import cv2

from preprocessing import hand_extractor
from threading import Event
from time import sleep

import RPS


def main():
    camera = cv2.VideoCapture(0)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)  # float width
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sg.theme('DarkAmber')

    # define the window layout

    layout = [[sg.Text('RPS GAME', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Text("User:"), sg.Image(filename='resourses/gui_images/background.png', key='image', size=(256, 256)),
               sg.Text(text="Computer"),
               sg.Image(filename='resourses/gui_images/background.png', key='-computerkey-', visible=True,
                        size=(256, 256))],
              [sg.Button("Start Game", size=(10, 1), font='Helvetica 14'),
               sg.Button('Start Camera', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14', ),
               sg.Button('Settings', size=(10, 1), font='Any 14', ),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14')],
              [sg.Text('Сложность:', size=(15, 1), justification='center', font='Helvetica 20'),
               sg.Text(text='Нормально', key='-settings-', font='Helvetica 20', )],
              [sg.Text(key="pobed", size=(30, 1), justification='center', font='Helvetica 20')]]

    # create the window and show it without the plot
    window = sg.Window('RPS GAME',
                       layout, size=(700, 450), location=(width, height), resizable=True)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #

    recording = False
    start_game = 0
    answer = 5


    while True:

        event, values = window.read(timeout=20)
        ret, frame = camera.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Start Game':

            try:
                recording = True

            except ValueError as e:
                window['pobed'].update('Подключите камеру')
                recording = False
                continue

            window['-computerkey-'].update(filename='resourses/gui_images/3.png', size=(256, 256))
            window.refresh()

            sleep(1)

            window['-computerkey-'].update(filename='resourses/gui_images/2.png', size=(256, 256))
            window.refresh()

            sleep(1)

            window['-computerkey-'].update(filename='resourses/gui_images/1.png', size=(256, 256))
            window.refresh()

            Event().wait(1.0)

            start_game = 1

        elif event == 'Start Camera':

            recording = True

        elif event == 'Stop':

            recording = False

        elif event == 'Settings':

            if RPS.settings == 'hard':

                RPS.settings = 'normal'
                window['-settings-'].update('Нормально')
                window.refresh()

            elif RPS.settings == 'normal':

                RPS.settings = 'hard'
                window['-settings-'].update('Непобедимо')
                window.refresh()

        if recording:

            try:

                hand_points_image, hand_image, hand_image_model, gesture = hand_extractor(frame, width, height)
                imgbytes = cv2.imencode('.png', hand_image)[1].tobytes()  # присваивание изображения
                window['image'].update(data=imgbytes)

            except cv2.error:

                window['pobed'].update('Подключите камеру')
                recording = False

        if start_game == 1:
            try:
                window['image'].update(data=imgbytes)
                # -------------------------------------------------------------------------------------------
                # transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
                # with torch.no_grad():
                #     prediction = model_fuzing_hand(model_hand(transform(hand_image_model).unsqueeze(0)),
                #                                    model_dot_hand(transform(hand_points_image).unsqueeze(0)))

                # tensor = torch.max(prediction.data.cpu(), 1)[1]
                # answer = tensor.item()
                # -------------------------------------------------------------------------------------------

            except:
                continue

            window['image'].update(data=imgbytes)
            if gesture in [1, 2, 0]:

                try:
                    recording = False

                    # вывод картинки с ответом игрока

                    if gesture == 0:

                        window['image'].update(filename='resourses/gui_images/rock.png', size=(256, 256))

                    elif gesture == 1:

                        window['image'].update(filename='resourses/gui_images/scissors.png', size=(256, 256))

                    elif gesture == 2:

                        window['image'].update(filename='resourses/gui_images/paper.png', size=(256, 256))

                    window.refresh()

                except ValueError as e:

                    RPS.range_str = f"[0, {len(RPS.Action) - 1}]"
                    print("Не правильное значение, выберите число между")
                    continue

                # вывод картинки с ответом компьютера
                pc_gesture = random.randint(0, 2)

                if pc_gesture == 0:

                    window['-computerkey-'].update(filename='resourses/gui_images/rock.png', size=(256, 256))

                elif pc_gesture == 1:

                    window['-computerkey-'].update(filename='resourses/gui_images/scissors.png', size=(256, 256))

                elif pc_gesture == 2:

                    window['-computerkey-'].update(filename='resourses/gui_images/paper.png', size=(256, 256))

                if gesture == pc_gesture:

                    window['pobed'].update('Ничья')

                elif gesture + 1 == pc_gesture or gesture - 2 == pc_gesture:

                    window['pobed'].update('Победа')

                else:

                    window['pobed'].update('Поражение')

                window.refresh()
                start_game = 0


if __name__ == "__main__":
    main()
