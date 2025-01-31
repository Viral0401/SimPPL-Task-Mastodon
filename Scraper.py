import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import time

driver = webdriver.Chrome()

base_url = "https://mastodon.social/explore"

def scroll_to_next_tweet(last_tweet):
    while True:
        tweets = driver.find_elements(By.CLASS_NAME, "status__content__text")
        if last_tweet == tweets[0]:
            print("No new tweets found. Scrolling.")
            driver.execute_script("window.scrollBy(0, 100);")
            time.sleep(2)
        else:
            driver.execute_script("window.scrollBy(0, 3000);")
            break
    
    last_tweet = tweets[-1]
    return last_tweet

def scrape_data(url):
    names = driver.find_elements(By.CLASS_NAME, "display-name__html")
    #for name in names:
        #print(name.text)

    likes = driver.find_elements(By.CLASS_NAME, "icon-button__counter")
    #for like in likes:
        #print(like.text)
        
    statuses = driver.find_elements(By.CLASS_NAME, "status__content__text")
    #for status in statuses:
        #print(status.text)

    with open(r'viral.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if file.tell() == 0:
            writer.writerow(['Name','Text','Replies','Boosts','Favourites'])
        
        for i in range(len(names)):
            writer.writerow([names[i].text, statuses[i].text, likes[i+0].text, likes[i+1].text, likes[i+2].text])

page_number = 1
url = base_url.format(page_number)
driver.get(url)
last_tweet = None

while page_number<1000:
    try:
        # Check if the page exists
        WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.CLASS_NAME, "icon-button__counter")))
        last_tweet = scroll_to_next_tweet(last_tweet)
        print(last_tweet)
        scrape_data(url)
        page_number += 1
    except StaleElementReferenceException:
        driver.refresh()
        time.sleep(2) 

driver.quit()