import os
import io
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ecgapp/model/ecg_model_final.h5')
model = load_model(MODEL_PATH)
class_labels = ['Abnormal_Heartbeat', 'History_of_MI', 'Myocardial_Infarction', 'Normal']

class_messages = {
    'Abnormal_Heartbeat': "This ECG suggests irregular heart rhythms. Please consult a cardiologist to investigate the cause and assess treatment needs.",
    'History_of_MI': "The pattern indicates a prior heart attack. Regular follow-up and lifestyle changes are strongly recommended.",
    'Myocardial_Infarction': "This may indicate an active or recent heart attack. Seek immediate medical attention if symptoms are present.",
    'Normal': "Your ECG appears within normal limits. No signs of cardiac abnormalities were detected."
}

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        img_path = os.path.join(settings.MEDIA_ROOT, img.name)

        with open(img_path, 'wb+') as dest:
            for chunk in img.chunks():
                dest.write(chunk)

        img_loaded = image.load_img(img_path, target_size=(384, 384))
        img_array = image.img_to_array(img_loaded) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx] * 100
        pred_class = class_labels[pred_idx]
        user_message = class_messages[pred_class]

        context = {
            'prediction': pred_class,
            'confidence': f"{confidence:.2f}%",
            'image_path': img.name,
            'media_url': settings.MEDIA_URL,
            'message': user_message,
            'breakdown': zip(class_labels, [f"{p*100:.2f}%" for p in preds[0]]),
            'breakdown_raw': list(zip(class_labels, preds[0] * 100)),
        }
        return render(request, 'predict.html', context)

    return render(request, 'predict.html')

def export_pdf(request):
    if request.method == 'POST':
        prediction = request.POST['prediction']
        confidence = request.POST['confidence']
        image_path = request.POST['image_path']
        message = class_messages.get(prediction, "")
        labels = request.POST.getlist('labels')
        probs = request.POST.getlist('probs')

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, "PulseCheck: ECG Classification Result")

        p.setFont("Helvetica", 12)
        p.drawString(100, 720, f"Predicted Class: {prediction}")
        p.drawString(100, 700, f"Confidence: {confidence}")
        p.drawString(100, 675, "Message:")
        text = p.beginText(120, 660)
        text.setFont("Helvetica", 11)
        for line in message.split('. '):
            text.textLine(line.strip() + '.')
        p.drawText(text)

        y = 610
        p.setFont("Helvetica-Bold", 12)
        p.drawString(100, y, "Class Confidence Breakdown:")
        y -= 20
        for label, prob in zip(labels, probs):
            p.setFont("Helvetica", 11)
            p.drawString(120, y, f"{label}: {prob}%")
            y -= 20

        ecg_path = os.path.join(settings.MEDIA_ROOT, image_path)
        if os.path.exists(ecg_path):
            img_reader = ImageReader(ecg_path)
            p.drawImage(img_reader, 100, y - 240, width=350, height=240, preserveAspectRatio=True)

        p.showPage()
        p.save()
        buffer.seek(0)
        return HttpResponse(buffer, content_type='application/pdf', headers={
            'Content-Disposition': 'attachment; filename="ecg_result.pdf"'
        })
