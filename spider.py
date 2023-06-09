from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

from bs4 import BeautifulSoup
import requests

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# 如果使用的是Chrome无头浏览器
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1200x600')  # Optional

driver = webdriver.Chrome(chrome_options=options)

for idxDoc in range(1,200):
    sUrl = f"https://h5.40dhjen.cn/catalogue?id={idxDoc}"
    response = requests.get(sUrl)
    #页面不存在跳过
    if response.status_code == 404:
        print(sUrl,"不存在")
        continue

    driver.get(response.url)
    driver.implicitly_wait(20) 

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "taro-view-core.catalogue-list-item.hydrated"))
    )
    taro_views = driver.find_elements(By.CSS_SELECTOR, 'taro-view-core.catalogue-list-item.hydrated')
    raro_size = len(taro_views)
    title = driver.title

    print("文件数量:",raro_size,"文章名：",title)

    for idx in range(0,raro_size):
        txt = taro_views[idx].text
        print(f"文件名：{txt}")

        taro_views[idx].click()

        driver.implicitly_wait(20)
        # 不是具体的页面跳过
        if "catalogue" in driver.current_url:
            break

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "taro-view-core.book-article-paragraph.hydrated"))
        )

        filename = f"{title}{idx:03d}{txt}.md" 
        # 获取新页面的内容
        new_html = driver.page_source
        new_soup = BeautifulSoup(new_html, 'html.parser')

        # 提取book-article-paragraph类的元素并写入文件
        paragraphs = new_soup.select('taro-view-core.book-article-paragraph.hydrated')
        with open(filename, 'w', encoding='utf-8') as file:
            for paragraph in paragraphs:
                file.write(paragraph.text + '\n')

        # 返回原页面
        driver.back()
        driver.implicitly_wait(20)
        taro_views = driver.find_elements(By.CSS_SELECTOR, 'taro-view-core.catalogue-list-item.hydrated')

driver.quit()
