from pages.login_page import LoginPage
from appium.webdriver.common.appiumby import AppiumBy

def test_positive_login(driver):
    login_page = LoginPage(driver)
    login_page.login("bob@example.com", "10203040")
    
    # Проверка появления заголовка "Products"
    label = driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().text("Products")')
    assert label.is_displayed()
