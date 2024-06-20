# If you need to import additional packages or classes, please import here.
def match_str(name_list, name):
    lenth = name.__len__()
    for i in range(lenth):
        sub_name = name[0:lenth - i]
        find_num = 0
        for book in name_list:
            idx = book.find(sub_name)
            print(idx)


def func():
    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    N = int(input())
    cmd_list = []
    book_list = []
    for i in range(N):
        cmd = input().strip().split()
        cmd_list.append(cmd)
    # print(cmd_list)

    # please finish the function body here.
    for cmd, name in cmd_list:
        if cmd == 'ADD':
            book_list.append(name)
            print('add ' + name)

        elif cmd == 'QUERY':
            print('query ' + name)
            match_str(book_list, name)

        else:
            print('delete ' + name)

    # please define the python3 output here. For example: print().


if __name__ == "__main__":
    func()
