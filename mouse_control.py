import pyautogui

screenWidth, screenHeight = pyautogui.size()
print(f"screenWidth: {screenWidth}, screenHeight: {screenHeight}")


def move_mouse(x, y):
    coef_x = (1 - x) if (1 - x) < 1 else 1
    coef_y = 1.2 * y if 1.2 * y < 1 else 1
    x = coef_x * screenWidth
    y = coef_y * screenHeight

    try:
        pyautogui.moveTo(x, y)
    except pyautogui.FailSafeException:
        pass

    currentMouseX, currentMouseY = pyautogui.position()
    print(f"currentMouseX: {currentMouseX}, currentMouseY: {currentMouseY}")

# pyautogui.move(-400, 100)
# pyautogui.doubleClick()  # https://pyautogui.readthedocs.io/en/latest/
