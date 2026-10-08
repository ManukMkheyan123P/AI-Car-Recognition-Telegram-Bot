
import os
import io
import pickle
import numpy as np

from pathlib import Path
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# ==============================
# 1. Загрузка модели
# ==============================

BASE_DIR = Path(__file__).resolve().parent

model = load_model(
    BASE_DIR / "car_model.keras",
    compile=False
)

with open(BASE_DIR / "class_names.pkl", "rb") as f:
    classes = pickle.load(f)

print("Тип классов:", type(classes))
print("Пример классов:", list(classes.items())[:5]
      if isinstance(classes, dict) else list(classes)[:5])

HEIGHT = model.input_shape[1]
WIDTH = model.input_shape[2]

# Важно: стандартный EfficientNetB0 ожидает
# пиксели 0–255, без деления на 255.
# Если при обучении ты делил на 255,
# установи значение True.
NORMALIZE_255 = False

# ==============================
# 2. Команда /start
# ==============================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "🚗 Привет!\n\n"
        "Я AI-бот для распознавания автомобилей.\n"
        "Отправь фотографию машины, "
        "и я попробую определить её модель! 🤖"
    )


# ==============================
# 3. Распознавание фотографии
# ==============================

async def predict_car(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "🔍 Анализирую фотографию..."
    )

    try:
        photo = update.message.photo[-1]

        telegram_file = await photo.get_file()

        image_bytes = await telegram_file.download_as_bytearray()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = ImageOps.fit(
            image,
            (WIDTH, HEIGHT)
        )

        image = np.array(
            image,
            dtype=np.float32
        )

        if NORMALIZE_255:
            image = image / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        predictions = model.predict(
            image,
            verbose=0
        )[0]

        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id]) * 100

        car_name = classes[class_id]

        # Топ-3 предсказания
        top_3 = np.argsort(predictions)[-3:][::-1]

        result = (
            f"🚘 Автомобиль: {car_name}\n"
            f"📊 Уверенность модели: {confidence:.2f}%\n\n"
            f"🏆 Топ-3 предсказания:\n"
        )

        for i, idx in enumerate(top_3, start=1):
            result += (
                f"{i}. {classes[int(idx)]} — "
                f"{predictions[idx] * 100:.2f}%\n"
            )

        await update.message.reply_text(result)

    except Exception as e:
        print("Prediction error:", e)

        await update.message.reply_text(
            "❌ Не удалось обработать фотографию."
        )


# ==============================
# 4. Запуск Telegram Bot
# ==============================

def main():

    TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

    app = Application.builder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            predict_car
        )
    )

    print("✅ Car Prediction Bot запущен!")

    app.run_polling()


if __name__ == "__main__":
    main()
