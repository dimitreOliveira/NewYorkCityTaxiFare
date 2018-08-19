import csv


def clean_data(input_data_path='data/train.csv', output_data_path='data/train_cleaned.csv'):
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            # remove header
            if count > 0:
                # only rows with non-null values
                if len(row) == 8:
                    try:
                        fare_amount = float(row[1])
                        pickup_longitude = float(row[3])
                        pickup_latitude = float(row[4])
                        dropoff_longitude = float(row[5])
                        dropoff_latitude = float(row[6])
                        passenger_count = float(row[7])
                        if ((-76 <= pickup_longitude <= -72) and (-76 <= dropoff_longitude <= -72) and
                                (38 <= pickup_latitude <= 42) and (38 <= dropoff_latitude <= 42) and
                                (1 <= passenger_count <= 6) and fare_amount > 0):
                            writer.writerow(row)
                    except:
                        pass
            count += 1


def split_data(input_data_path, train_data_path, validation_data_path, ratio=30):
    with open(input_data_path, 'r') as inp, open(train_data_path, 'w') as out1, open(validation_data_path, 'w') as out2:
        writer1 = csv.writer(out1)
        writer2 = csv.writer(out2)
        count = 0
        for row in csv.reader(inp):
            if len(row) > 0:
                if count % ratio == 0:
                    writer2.writerow(row)
                else:
                    writer1.writerow(row)
                count += 1


def pre_process_data(input_data_path='data/train_cleaned.csv', output_data_path='data/train_processed.csv'):
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if len(row) > 0:
                pickup_datetime = row[2]
                year = pickup_datetime[:4]
                month = pickup_datetime[5:7]
                day = pickup_datetime[8:10]
                hour = pickup_datetime[11:13]
                row.append(year)
                row.append(month)
                row.append(day)
                row.append(hour)
                writer.writerow(row)
