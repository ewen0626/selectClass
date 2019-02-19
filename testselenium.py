# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from selenium import webdriver
import selenium.webdriver.support.ui as ui
browser = webdriver.Chrome()

browser.get('http://course.uch.edu.tw/stdsel/System_D/signform.asp')

element = browser.find_elements_by_class_name("stu")[1].click() #進入選課系統

wait = ui.WebDriverWait(browser,10)
wait.until(lambda browser: browser.find_elements_by_name("fm1"))

element = browser.find_elements_by_name("fm1")

element = browser.find_element_by_name("txtaccount")

element.send_keys("M10711004")

element = browser.find_element_by_name("txtpassword")
element.send_keys("yuda39429")

#browser.find_elements_by_name("txtaccount").send_keys("M10711004")

#browser.find_elements_by_name("txtpassword")[1].send_keys("yuda39429")

