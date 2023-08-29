import re
def cleaner(message : str) -> str:

    #message = message.lower
    print(type(message))

    #removes reference of for (#12345) often found at the end of commit messages
    pattern = r'\(#\d+\)'
    message = re.sub(pattern, "", message)
    
    return message
