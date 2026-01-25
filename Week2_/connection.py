import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",        # or your server IP
    user="your_username",
    password="your_password",
    database="your_database"
)

print("Connected to MySQL!")

# Create a cursor
cursor = conn.cursor()


# Example query
cursor.execute("SELECT * FROM your_table")

# Fetch all results
rows = cursor.fetchall()

for row in rows:
    print(row)

# Close everything
cursor.close()
conn.close()
