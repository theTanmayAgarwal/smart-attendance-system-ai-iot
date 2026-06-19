import csv
import os
from datetime import datetime

CSV_FILE = "database/attendance.csv"


def mark_attendance(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")

    time = now.strftime("%H:%M:%S")

    records = []

    if os.path.exists(CSV_FILE):

        with open(CSV_FILE, "r") as file:

            reader = csv.reader(file)

            records = list(reader)

    for row in records:

        if len(row) >= 2:

            if row[0] == name and row[1] == date:

                return False

    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([name, date, time])

    return True