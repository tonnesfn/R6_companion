import os
import sys
import ctypes
from ctypes import wintypes
import win32con
from PIL import ImageGrab, Image, ImageEnhance, ImageOps
import pytesseract
from time import gmtime, strftime
import r6sapi as api
import asyncio
import requests
import processScreenshot

byref = ctypes.byref
user32 = ctypes.windll.user32
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'


HOTKEYS = {
    1: (win32con.VK_INSERT, None),
    2: (win32con.VK_F4, None)
}

def lookup_player(username):
    r = requests.get("https://api.r6stats.com/api/v1/players/%s?platform=uplay" % username)
    if r.status_code == 200:
        return r.json()
    else:
        return {"player":
                    {"username": "notfound",
                     "stats": {"progression": {"level": "N/A"},
                               "ranked": {"playtime": 36000,
                                          "kd": 0.0,
                                          "wlr": 0.0
                                          }
                               }
                     }
                }

def print_player(player, raw):
    username = player["player"]["username"]
    playtime = player["player"]["stats"]["ranked"]["playtime"] / 60 / 60
    kdr = player["player"]["stats"]["ranked"]["kd"]
    wlr = player["player"]["stats"]["ranked"]["wlr"]
    level = player["player"]["stats"]["progression"]["level"]
    print("""
    %s (%s):
        kd: %s
        wlr: %s
        playtime: %s
        level: %s
    """ % (username, raw, kdr, wlr, playtime, level))


def handle_win_f3 ():
    print("hotkey f3 pressed, screenshot starting")
    ImageGrab.grab().save("screen_capture.jpg", "JPEG")
    width, height = (2560, 1440)
    top_team = ImageGrab.grab(bbox=(width*0.18, height*0.31, width*0.38, height*0.55))
    top_team.save("top_%s.jpg"%strftime("%Y_%m_%d_%H%M%S", gmtime()), "JPEG")
    top_1 = ImageGrab.grab(bbox=(width * 0.18, height * 0.31, width * 0.38, height * 0.35))
    top_2 = ImageGrab.grab(bbox=(width * 0.18, height * 0.36, width * 0.38, height * 0.40))
    top_3 = ImageGrab.grab(bbox=(width * 0.18, height * 0.41, width * 0.38, height * 0.45))
    top_4 = ImageGrab.grab(bbox=(width * 0.18, height * 0.46, width * 0.38, height * 0.50))
    top_5 = ImageGrab.grab(bbox=(width * 0.18, height * 0.51, width * 0.38, height * 0.55))
    top = []
    current = 0.31
    player_height = 0.04
    offset = 0.01
    for x in range(5):
        print(current)
        print(current+player_height)
        current += player_height + offset
    top_1.save("test1.jpg", "JPEG")
    top_2.save("test2.jpg", "JPEG")
    top_3.save("test3.jpg", "JPEG")
    top_4.save("test4.jpg", "JPEG")
    top_5.save("test5.jpg", "JPEG")
    top_5alt = ImageOps.posterize(top_5, 4)
    top_5altalt = ImageOps.invert(top_5)
    top_5altalt.save("test5altalt.jpg", "JPEG")
    top_5alt.save("test5alt.jpg", "JPEG")
    print(pytesseract.image_to_string(top_1))
    print(pytesseract.image_to_string(top_2))
    print(pytesseract.image_to_string(top_3))
    print(pytesseract.image_to_string(top_4))
    print("regular")
    print(pytesseract.image_to_string(top_5))
    print("alt")
    print(pytesseract.image_to_string(top_5alt))
    print("altalt")
    print(pytesseract.image_to_string(top_5altalt))
    bottom_team = ImageGrab.grab(bbox=(width*0.18, height*0.65, width*0.38, height*0.90))
    #processScreenshot.process_screenshot(bottom_team.convert('L'))
    con = ImageEnhance.Contrast(top_team).enhance(2)
    con.save("con_%s.jpg" %strftime("%Y_%m_%d_%H%M%S", gmtime()), "JPEG")
    bottom_team.save("bottom_%s.jpg" %strftime("%Y_%m_%d_%H%M%S", gmtime()), "JPEG")
    top = pytesseract.image_to_string(top_team)
    for line in top.split("\n"):
        if len(line.strip()) > 0:
            player = lookup_player(line.strip())
            print_player(player, line.strip())

    bottom = pytesseract.image_to_string(bottom_team)
    for line in bottom.split("\n"):
        if len(line.strip()) > 0:
            player = lookup_player(line.strip())
            print_player(player, line.strip())
    print("screenshot done")


def handle_win_f4 ():
    player = lookup_player("crispiNor")
    username = player["player"]["username"]
    playtime = player["player"]["stats"]["ranked"]["playtime"]/60/60
    kdr = player["player"]["stats"]["ranked"]["kd"]
    wlr = player["player"]["stats"]["ranked"]["wlr"]
    level = player["player"]["stats"]["progression"]["level"]
    print("""
    %s:
        kd: %s
        wlr: %s
        playtime: %s
        level: %s
    """ %(username, kdr, wlr, playtime, level))
    print("hotkey f4 pressed")


HOTKEY_ACTIONS = {
    1: handle_win_f3,
    2: handle_win_f4
}

def main():
    for id, (vk, modifiers) in HOTKEYS.items():
        print("Registering id", id, "for key", vk)
        if not user32.RegisterHotKey(None, id, modifiers, vk):
            print("Unable to register id", id)
    try:
        msg = wintypes.MSG()
        while user32.GetMessageA(byref(msg), None, 0, 0) != 0:
            if msg.message == win32con.WM_HOTKEY:
                action_to_take = HOTKEY_ACTIONS.get(msg.wParam)
            if action_to_take:
                action_to_take()
        user32.TranslateMessage(byref(msg))
        user32.DispatchMessageA(byref(msg))
    finally:
        for id in HOTKEYS.keys():
            user32.UnregisterHotKey(None, id)

if __name__ == "__main__":
    main()
    print("done with life")