import os
import sys
import ctypes
from ctypes import wintypes
import win32con
from PIL import ImageGrab, Image, ImageEnhance, ImageOps
import pytesseract
from time import gmtime, strftime
import requests
import processScreenshot

byref = ctypes.byref
user32 = ctypes.windll.user32
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
script_dir = os.path.dirname(__file__)

HOTKEYS = {
    1: (win32con.VK_F3, None),
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
    width, height = (2560, 1440)
    with open(os.path.join(os.path.curdir, 'screenshot_examples', "screenshot_%s.jpg" % strftime("%Y_%m_%d_%H%M%S", gmtime())), "w+") as f:
        screen = ImageGrab.grab()
        screen.save(f)
        top_names, bottom_names = processScreenshot.screenshot_capture.get_names()

        top_team = processScreenshot.get_nicks(top_names, False)
        bottom_team = processScreenshot.get_nicks(bottom_names, False)

        for player in top_team:
            for alternative_name in player:
                p = lookup_player(alternative_name)
                if p["player"]["username"] == "notfound":
                    continue
                else:
                    print_player(alternative_name, p)
                    break
        for player in bottom_team:
            for alternative_name in player:
                p = lookup_player(alternative_name)
                if p["player"]["username"] == "notfound":
                    continue
                else:
                    print_player(alternative_name, p)
                    break


    comment = """
    top_team = ImageGrab.grab(bbox=(width*0.18, height*0.31, width*0.38, height*0.55)).convert('L')
    with open(os.path.join(os.path.curdir, 'screenshot_examples', "top_%s.jpg" % strftime("%Y_%m_%d_%H%M%S", gmtime())), "w+") as f:
        top_team.save(f, "JPEG")
    bottom_team = ImageGrab.grab(bbox=(width*0.18, height*0.66, width*0.38, height*0.90)).convert('L')
    with open(os.path.join(os.path.curdir, 'screenshot_examples', "bottom_%s.jpg" % strftime("%Y_%m_%d_%H%M%S", gmtime())), "w+") as f:
        bottom_team.save(f, "JPEG")

    top = pytesseract.image_to_string(top_team, config="-c tessedit_char_whitelist=.-_1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM")
    for line in top.split("\n"):
        if len(line.strip()) > 0:
            player = lookup_player(line.strip())
            print_player(player, line.strip())
    bottom = pytesseract.image_to_string(bottom_team)
    for line in bottom.split("\n"):
        if len(line.strip()) > 0:
            player = lookup_player(line.strip())
            print_player(player, line.strip())
    """

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