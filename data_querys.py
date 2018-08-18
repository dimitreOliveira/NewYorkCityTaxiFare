import csv


def remove_header(input_data_path='data/train.csv', output_data_path='data/tf_train.csv'):
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            if count > 0:
                writer.writerow(row)
            count += 1


def split_data(input_data_path, train_data_path, validation_data_path, ratio=20):
    with open(input_data_path, 'r') as inp, open(train_data_path, 'w') as out1, open(validation_data_path, 'w') as out2:
        writer1 = csv.writer(out1)
        writer2 = csv.writer(out2)
        count = 0
        for row in csv.reader(inp):
            if count % ratio == 0:
                writer2.writerow(row)
            else:
                writer1.writerow(row)
            count += 1


def clean_data(input_data_path, output_data_path):
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if len(row) > 0:
                try:
                    fare_amount = float(row[1])
                    pickup_longitude = float(row[3])
                    pickup_latitude = float(row[4])
                    dropoff_longitude = float(row[5])
                    dropoff_latitude = float(row[6])
                    passenger_count = int(row[7])
                    if ((-76 <= pickup_longitude <= -72) and (-76 <= dropoff_longitude <= -72) and
                            (38 <= pickup_latitude <= 42) and (38 <= dropoff_latitude <= 42) and passenger_count > 0 and
                                fare_amount > 0):
                        writer.writerow(row)
                except:
                    pass
