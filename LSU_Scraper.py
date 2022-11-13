from bs4 import BeautifulSoup
import requests

result = []
import csv

url = 'https://www.espn.com/college-football/team/schedule/_/id/99/season/'
nattyURL = 'https://en.wikipedia.org/wiki/List_of_LSU_Tigers_football_seasons'
year = 2003

nattyPage = requests.get(nattyURL)
soup = BeautifulSoup(nattyPage.content, 'html.parser')
wins = [x.get_text() for x in soup.find_all('tr', attrs={'style':'background:#fc6'})]
for x in range(len(wins)):
    currentWin = wins[x]
    res = "".join([ele for ele in currentWin if ele.isdigit()])
    result = ''
    for y in res:
        if len(result) < 4:
            result = result + y
    wins[x] = result

SEC_Wins = [x.get_text() for x in soup.find_all('tr', attrs={'style':'background:#ff9'})]
for x in range(len(SEC_Wins)):
    currentWin = SEC_Wins[x]
    res = "".join([ele for ele in currentWin if ele.isdigit()])
    result = ''
    for y in res:
        if len(result) < 4:
            result = result + y
    SEC_Wins[x] = result

records = []
for x in range (0, 20, 1):
    currentYear = year + x
    currentURL = url + str(currentYear)
    page = requests.get(currentURL)
    soup = BeautifulSoup(page.content, 'html.parser')
    winResults = [x.get_text() for x in soup.find_all('span', class_="fw-bold clr-positive")]
    lossResults = [y.get_text() for y in soup.find_all('span', class_="fw-bold clr-negative")]

    gamesPlayed = (len(winResults) + len(lossResults))
    percentage = round((len(winResults) / gamesPlayed), 2)
    natty = False
    SEC_Win = False

    for i in range(len(wins)):
        if str(currentYear) == wins[i]:
            natty = True
            SEC_Win = True
        if str(currentYear) == SEC_Wins[i]:
            SEC_Win = True
    record = (currentYear, percentage, len(winResults), len(lossResults), SEC_Win, natty)
    records.append(record)

csv_file = "Records.csv"

with open(csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['year', 'win percentage', 'wins', 'losses', 'SEC Win?', 'Natty?'])
    for row in records:
        writer.writerow(row)

