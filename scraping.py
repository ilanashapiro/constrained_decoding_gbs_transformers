import requests
from bs4 import BeautifulSoup
import re
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# Step 1: Fetch the webpage
url = "https://www.gutenberg.org/cache/epub/57333/pg57333-images.html"
response = requests.get(url)

# Step 2: Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Step 3: Extract visible text (excluding scripts & styles)
for script in soup(["script", "style"]):
    script.extract()  # Remove unnecessary elements

text = soup.get_text(separator="\n", strip=True)  # Extract cleaned text
cleaned_text = re.sub(r"\n\s*\n", "\n", text)  # Remove extra newlines
cleaned_text = re.sub(r"[^a-zA-Z0-9\s\-,.!?]", "", cleaned_text)  # Keep only words & punctuation
cleaned_text = re.sub(r" +", " ", cleaned_text)  # Remove excessive spaces but keep \n intact
trigger = "HOSPITAL" #start scraping from this line in the website
extracted_text = trigger + cleaned_text.split(trigger, 1)[1] 
extracted_text = extracted_text.split("END OF THE PROJECT", 1)[0] #remove the end of the book
with open(dir_path+"/datasets/scraped_chekhov.txt", "w") as file:
    file.write(extracted_text)  # Save extracted text to a file

url = "https://www.gutenberg.org/cache/epub/7986/pg7986-images.html"
response = requests.get(url)

# Step 2: Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Step 3: Extract visible text (excluding scripts & styles)
for script in soup(["script", "style"]):
    script.extract()  # Remove unnecessary elements

text = soup.get_text(separator="\n", strip=True)  # Extract cleaned text
cleaned_text = re.sub(r"\n\s*\n", "\n", text)  # Remove extra newlines
cleaned_text = re.sub(r"[^a-zA-Z0-9\s\-,.!?]", "", cleaned_text)  # Keep only words & punctuation
cleaned_text = re.sub(r" +", " ", cleaned_text)  # Remove excessive spaces but keep \n intact
trigger = "The action takes place in one of the provinces of Southern Russia" #start scraping from this line in the website
extracted_text = "\n"+trigger + cleaned_text.split(trigger, 1)[1] 
extracted_text = extracted_text.split("END OF THE PROJECT", 1)[0] #remove the end of the book
with open(dir_path+"/datasets/scraped_chekhov.txt", "a") as file:
    file.write(extracted_text)  # Save extracted text to a file
