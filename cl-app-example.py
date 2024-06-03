import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser(description='To-do parser')
    parser.add_argument('-a', "--add", metavar="", help="Add new task")
    parser.add_argument('-l', "--list", action = "store_true", help='List tasks')
    parser.add_argument('-r', "--remove", metavar="", help='Remove task by index')
    return parser

def add_task(task):
    with open('tasks.txt', 'a') as f:
        f.write(task + '\n')

def list_tasks():
    if os.path.exists('tasks.txt'):
        with open('tasks.txt', 'r') as f:
            tasks = f.readlines()
            for i, task in enumerate(tasks):
                print(f'{i + 1}. {task}', end='')
    else:
        print('No tasks')

def remove_task(index):
    if os.path.exists("tasks.txt"):
        with open("tasks.txt", "r") as file:
            tasks = file.readlines()
        with open("tasks.txt", "w") as file:
            for i, task in enumerate(tasks, start=1):
                if i != index:
                    file.write(task)
        print("Task removed successfully.")
    else:
        print("No tasks found.")

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.add:
        add_task(args.add)
    elif args.list:
        list_tasks()
    elif args.remove:
        remove_task(int(args.remove))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()