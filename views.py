# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from datetime import datetime
# import logging

# logger = logging.getLogger(__name__)

# @csrf_exempt
# def set_reminder(request):
#     if request.method == 'POST':
#         message = request.POST.get('message')
#         time = request.POST.get('time')
        
#         logger.info(f"Received message: {message}")
#         logger.info(f"Received time: {time}")
        
#         try:
#             reminder_time = datetime.strptime(time, '%Y-%m-%d %H:%M')
#             # Here you would save the reminder to the database or schedule it
#             return JsonResponse({'message': 'Reminder set successfully!'})
#         except ValueError:
#             return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD HH:MM:SS.'}, status=400)
#     return JsonResponse({'error': 'Invalid request method.'}, status=405)
