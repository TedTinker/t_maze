import keyboard

def is_q_pressed():
    q = keyboard.is_pressed("q")
    return(q)

print(is_q_pressed(), end='')