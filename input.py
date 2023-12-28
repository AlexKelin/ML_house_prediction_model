from main import rf_model
import pandas as pd


def predict_house_price(model, rooms, landsize, year_built):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'Rooms': [rooms],
        'Landsize': [landsize],
        'YearBuilt': [year_built]
    })

    # Use the model to make a prediction
    try:
        predicted_price = model.predict(input_data)[0]  # [0] to extract single value from array
        return int(predicted_price)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


def get_user_input():
    try:
        rooms = int(input("Please input number of rooms: "))
        landsize = float(input("Please input land size: "))
        year_built = int(input("Please input year built: "))
        return rooms, landsize, year_built
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None, None, None


# Get user input
rooms, landsize, year_built = get_user_input()

# Make prediction if valid input is provided
if rooms is not None and landsize is not None and year_built is not None:
    predicted_price = predict_house_price(rf_model, rooms, landsize, year_built)
    if predicted_price is not None:
        formatted_price = "${:,}".format(predicted_price)
        print(
            f"Predicted price for house with {rooms} rooms, landsize of {landsize}, and built in {year_built}: {formatted_price}")
