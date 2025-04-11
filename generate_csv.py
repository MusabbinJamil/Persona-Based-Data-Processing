import csv
from faker import Faker

fake = Faker()

# Define the output CSV filepath and field names (updated)
output_csv = "raw_data.csv"
fieldnames = ["id", "text", "name", "email", "address", "created_at"]
print(f"Generating CSV file at: {output_csv}")

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(10000):
        writer.writerow({
            "id": i + 1,
            "text": fake.text(max_nb_chars=200),
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address().replace("\n", ", "),
            "created_at": fake.date_time_between(start_date="-2y", end_date="now").strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"Generated record {i + 1}")

print(f"CSV file with 10000 records has been generated at: {output_csv}")