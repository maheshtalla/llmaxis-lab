#open file
# relative path: input.txt
#absolute path: C:\Users\mahes\OneDrive\Documents\github-repo\llmaxis-lab\input.txt
file = open("input.txt", "r") 

#store content of file in a variable
content = file.read()   

#close the file
file.close()    

print(content)

#writing to a file
#open file in write mode
file = open("input.txt", "a+") 

#write content to file
file.write("\nHello, adding some content to the file.") 
file.write("\nThis is a new line added to the file.")  

#store content of file in a variable
content = file.read()   

#close the file
file.close()    

print(content)