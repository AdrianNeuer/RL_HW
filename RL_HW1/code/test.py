def write_file(action):
    with open('imgs/test.txt', 'a') as f:
        f.write(str(action) + "\n")


if __name__ == "__main__":
    action_list = [1, 2, 3, 4, 6, 6]
    for action in action_list:
        write_file(action)
