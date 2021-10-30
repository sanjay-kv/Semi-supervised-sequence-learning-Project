# product name
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import csv
import pandas as pd

## One way to load chrome webdirver
#from webdriver_manager.chrome import ChromeDriverManager
#driver = webdriver.Chrome(ChromeDriverManager().install())

## another way to load chrome webdriver
path = '/Users/mohammedrizwan/Downloads/chromedriver'
driver = webdriver.Chrome(path)

def product_listing(txt):

    driver.get("https://www.amazon.in/")
    driver.implicitly_wait(2)
    search = driver.find_element_by_id('twotabsearchtextbox').send_keys(txt)
    driver.implicitly_wait(2)
    search_button = driver.find_element_by_id('nav-search-submit-button').click()
    driver.implicitly_wait(5)

    # product_name = []
    items = WebDriverWait(driver,10).until(EC.presence_of_all_elements_located((By.XPATH, '//a[@class="a-link-normal a-text-normal"]')))

    # j=0
    # print(f"length of {txt} {len(items)}")
    for item in items:
        name_list.append(item.text)

    driver.implicitly_wait(5)
    c1 = driver.find_element_by_class_name("a-pagination")
    c2 = c1.text
    c3 = c2.splitlines()
    num_of_pg = c3[-2]

    for i in range(int(num_of_pg)-5):
        print(i)
        items = WebDriverWait(driver,10).until(EC.presence_of_all_elements_located((By.XPATH, '//a[@class="a-link-normal a-text-normal"]')))
        for item in items:
            name_list.append(item.text)
        link = driver.find_element_by_class_name("a-section.a-spacing-none.a-padding-base")
        next_lin = link.find_element_by_class_name("a-last").find_element_by_tag_name("a").get_attribute("href")
        driver.get(next_lin)
        driver.implicitly_wait(2)

    # return product_name

#names = product_listing("trimmer")

# names = ['Laptop', 'Phones', 'Printers', 'Desktops', 'All in one pc', 'Adhesives', 'Mobile toilets', 'Art and craft', 'Smart TV', 'Air freshneres']
# for i in names:
#     name_list = product_listing(i)
#     file_name = i + '.txt'
#     with open(file_name, 'w') as f:
#         for item in name_list:
#             f.write(f"{item}\n")

# driver.get("https://www.amazon.in/")
# driver.implicitly_wait(2)
# search = driver.find_element_by_id('twotabsearchtextbox').send_keys("laptop")
# driver.implicitly_wait(2)
# search_button = driver.find_element_by_id('nav-search-submit-button').click()
# driver.implicitly_wait(5)
# c1 = driver.find_element_by_class_name("a-pagination")
# c2 = c1.text
# c3 = c2.splitlines()
# num_of_pg = c3[-2]
# print(type(c3))
# print(c3)
# print(num_of_pg)
# i=1
# product_name = []

# link = driver.find_element_by_class_name("a-section.a-spacing-none.a-padding-base")
# next_lin = link.find_element_by_class_name("a-last").find_element_by_tag_name("a").get_attribute("href")
# driver.get(next_lin)
# driver.implicitly_wait(5)


names = ['Laptop', 'Phones', 'Printers', 'Desktops', 'Monitors', 'Mouse', 'Pendrive', 'Earphones', 'Smart TV', 'Power banks']
name_list = []
for i in names:
    product_listing(i)
df=pd.DataFrame(name_list)
df.to_csv('./prod_listings.csv')
print(df)
driver.quit()



    # name = item.find_element_by_xpath('//span[@class="a-size-medium a-color-base a-text-normal"]')
    # product_name.append(name.text)
    # j += 1
    # if(j==2):
    #     break

# print("-----------------------------------")
# print(f"Number of products listed as {txt} is {len(product_name)}")
# print("-----------------------------------")
# for i in product_name:
#     print(f"{i}\n")




# # Step - a : Remove blank rows if any.
# Corpus['text'].dropna(inplace=True)
# # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# Corpus['text'] = [entry.lower() for entry in Corpus['text']]
# # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
# Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
# # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# tag_map = defaultdict(lambda : wn.NOUN)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV
# for index,entry in enumerate(Corpus['text']):
#     # Declaring Empty List to store the words that follow the rules for this step
#     Final_words = []
#     # Initializing WordNetLemmatizer()
#     word_Lemmatized = WordNetLemmatizer()
#     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#     for word, tag in pos_tag(entry):
#         # Below condition is to check for Stop words and consider only alphabets
#         if word not in stopwords.words('english') and word.isalpha():
#             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#             Final_words.append(word_Final)
#     # The final processed set of words for each iteration will be stored in 'text_final'
#     Corpus.loc[index,'text_final'] = str(Final_words)

