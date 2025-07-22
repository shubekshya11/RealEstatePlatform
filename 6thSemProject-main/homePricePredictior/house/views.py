from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.urls import reverse
from django.db import IntegrityError
from django.views.decorators.http import require_POST
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Property, ContactMessage, PropertyImage, Message,User, HousePricePrediction, Wishlist
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .utils import recommend_similar_properties
from house.models import Property
import re
import os
import numpy as np
import pandas as pd
import pickle
from django.http import HttpResponse
import json
from joblib import load
from pathlib import Path
from django.core.mail import EmailMessage
from house.ml_models.svm_model import SVR
from house.ml_models.decision_tree import DecisionTreeRegressor
from .feature_engineering import engineer_features  # Import the feature engineering function

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


BASE_DIR = Path(__file__).resolve().parent
print("Current directory:", os.getcwd())
print("File location:", BASE_DIR)
print("Model directory:", os.path.join(BASE_DIR, 'ml_models', 'saved_models'))
def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not re.search(r'[a-zA-Z]', password) or not re.search(r'\d', password):
        return False, "Password must contain both letters and numbers."
    
    return True, ""

def is_admin(user):
    return user.is_staff or user.is_superuser

def signup_view(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        is_valid, error_message = validate_password(password)
        if not is_valid:
            messages.error(request, error_message)
            return redirect('signup')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use.")
            return redirect('signup')

        user = User.objects.create_user(
            email=email,
            password=password,
            first_name=name,
            phone=phone,
            is_active=False  # Set to False until email is verified
        )

        # Generate email verification token
        current_site = get_current_site(request)
        mail_subject = 'Activate your account'
        message = render_to_string('email/verification_email.html', {
            'user': user,
            'domain': '127.0.0.1:8000',  # Use explicit domain
            'uid': urlsafe_base64_encode(force_bytes(user.pk)),
            'token': default_token_generator.make_token(user),
            'protocol': 'http',  # Use http for local development
        })
        
        email = EmailMessage(
            mail_subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [email]
        )
        email.send()

        messages.success(request, 'Please check your email to verify your account before logging in.')
        return redirect('login')
    
    return render(request, 'signup.html')


def activate_view(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, 'Thank you for confirming your email. You can now log in.')
        return redirect('login')
    else:
        messages.error(request, 'The activation link is invalid or expired.')
        return redirect('signup')


def login_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Your account is not activated. Please check your email.')
                return redirect('login')
        else:
            messages.error(request, 'Invalid email or password')
            return redirect('login')

    return render(request, 'login.html')


# Logout view
@login_required
def logout_view(request):
    logout(request)
    next_url = request.GET.get('next', 'login')  
    return redirect(next_url)

# Password Reset View
def password_reset_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        user = User.objects.filter(email=email).first()
        
        if user:
            current_site = get_current_site(request)
            mail_subject = 'Reset your password'
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = f"http://{current_site.domain}{reverse('password_reset_confirm_view', kwargs={'uidb64': uid, 'token': token})}"
            message = f"Hi {user.username},\n\nClick the link below to reset your password:\n{reset_link}"
            send_mail(mail_subject, message, 'noreply@example.com', [email])
            
            messages.success(request, "A password reset link has been sent to your email.")
        else:
            messages.info(request, "If an account with this email exists, a reset link has been sent.")
        
        return render(request, 'password_reset.html')
    return render(request, 'password_reset.html')

def password_reset_confirm_view(request, uidb64, token):
    print(f"Received uidb64: {uidb64}, token: {token}") 
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        print(f"Decoded UID: {uid}")  # Debug
        user = User.objects.get(pk=uid)
        print(f"User: {user}")  # Debug
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        print(f"Error: {e}") # Debug
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST['new_password']
            user.set_password(new_password)
            user.save()
            messages.success(request, "Your password has been successfully reset. You can now log in.")
            return redirect('login')
        return render(request, 'password_reset_confirm.html', {'valid_link': True, 'uidb64': uidb64, 'token': token })
    else:
        messages.error(request, "The password reset link is invalid or has expired.")
        return render(request, 'password_reset_confirm.html', {'valid_link': False})


def home(request):
    query_title = request.GET.get('title', '')
    query_city = request.GET.get('city', '')
    
    properties = Property.objects.filter(is_approved=True)

    if query_title:
        properties = properties.filter(title__icontains=query_title)
    
    if query_city:
        properties = properties.filter(city__icontains=query_city)
    
    properties = properties.order_by('-created_at')[:4]
    
    return render(request, 'home.html', {
        'properties': properties,
        'query_title': query_title,
        'query_city': query_city,
    })

def user_dashboard(request):
    user_properties = Property.objects.filter(seller=request.user)  # Get properties listed by the logged-in user
    
    context = {
        'user_properties': user_properties
    }
    return render(request, 'user_dashboard.html', context)
    # return render(request, "user_dashboard.html")

def admin_dashboard(request):
    datasets_dir = os.path.join('media', 'datasets')
    dataset_files = []
    
    if os.path.exists(datasets_dir):
        for file in os.listdir(datasets_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(datasets_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    preview = df.head(2).to_html(classes='table table-striped table-bordered', index=False)
                    dataset_files.append({
                        'name': file,
                        'path': file_path,
                        'preview': preview,
                        'total_rows': len(df)
                    })
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")

    pending_properties = Property.objects.filter(is_approved=False, decline_reason__isnull=True)
    
    context = {
        'properties': pending_properties,
        'datasets': dataset_files,
    }
    return render(request, "admin_dashboard.html", context)

def view_dataset(request, filename):
    try:
        file_path = os.path.join('media', 'datasets', filename)
        if not os.path.exists(file_path):
            messages.error(request, f"Dataset file {filename} not found.")
            return redirect('admin_dashboard')
            
        df = pd.read_csv(file_path)
        dataset_html = df.to_html(
            classes='table table-striped table-hover',
            index=False,
            escape=False
        )
        
        context = {
            'dataset_name': filename,
            'dataset_html': dataset_html,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        return render(request, 'view_dataset.html', context)
        
    except Exception as e:
        messages.error(request, f"Error viewing dataset: {str(e)}")
        return redirect('admin_dashboard')
    
def buyer(request):
    properties = Property.objects.filter(is_approved=True).order_by('-created_at')
    return render(request, 'buyer.html', {
        'properties': properties
    })
@user_passes_test(lambda u: u.is_superuser)
def approve_property(request, property_id):
    """Admin approves a property"""
    property_obj = Property.objects.get(id=property_id)
    property_obj.is_approved = True
    property_obj.save()

    # Send approval email to seller
    current_site = get_current_site(request)
    mail_subject = "Your Property Listing has been Approved!"
    message = render_to_string('email/property_listing_notification.html', {
        'user': property_obj.seller,
        'property': property_obj,
        'status': 'approved',
        'domain': current_site.domain,
        'protocol': 'https' if request.is_secure() else 'http',
    })
    
    email = EmailMessage(
        mail_subject,
        message,
        settings.DEFAULT_FROM_EMAIL,
        [property_obj.seller.email]
    )
    email.send()

    messages.success(request, "Property approved successfully and email sent to the user.")
    return redirect("admin_dashboard")

@user_passes_test(lambda u: u.is_superuser)
def decline_property(request, property_id):
    property = get_object_or_404(Property, id=property_id)
    
    if request.method == 'POST':
        decline_reason = request.POST.get('decline_reason')
        
        # Update property status
        property.is_approved = False
        property.decline_reason = decline_reason
        property.save()
        
        # Send decline email to seller
        current_site = get_current_site(request)
        mail_subject = "Your Property Listing has been Declined"
        message = render_to_string('email/property_listing_notification.html', {
            'user': property.seller,
            'property': property,
            'status': 'declined',
            'domain': current_site.domain,
            'protocol': 'https' if request.is_secure() else 'http',
        })
        
        email = EmailMessage(
            mail_subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [property.seller.email]
        )
        email.send()
        
        messages.success(request, "Property declined successfully, and an email has been sent to the user.")
        return redirect("admin_dashboard")
    
    return redirect('admin_dashboard')

@user_passes_test(lambda u: u.is_superuser)
def pending_properties(request):
    """List properties pending approval"""
    properties = Property.objects.filter(is_approved=False)  # Unapproved properties
    return render(request, "dashboard.html", {"properties": properties})

@login_required
def seller_view(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        city = request.POST.get('city')
        area = request.POST.get('area')
        bedrooms = request.POST.get('bedrooms')
        bathrooms = request.POST.get('bathrooms')
        stories = request.POST.get('stories')
        mainroad = request.POST.get('mainroad') == 'yes'
        guestroom = request.POST.get('guestroom') == 'yes'
        basement = request.POST.get('basement') == 'yes'
        hotwaterheating = request.POST.get('hotwaterheating') == 'yes'
        airconditioning = request.POST.get('airconditioning') == 'yes'
        parking = request.POST.get('parking')
        furnishingstatus = request.POST.get('furnishingstatus')
        price = request.POST.get('price')
        property_images = request.FILES.getlist('property_images')
        try:
            if not title or not city or not area or not bedrooms or not bathrooms or not stories or not price:
                raise ValidationError('Please fill out all required fields.')
            
    
            property = Property(
                title=title,
                city=city,
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                stories=stories,
                mainroad=mainroad,
                guestroom=guestroom,
                basement=basement,
                hotwaterheating=hotwaterheating,
                airconditioning=airconditioning,
                parking=parking,
                furnishingstatus=furnishingstatus,
                price=price,
                seller=request.user,  
                sale_status='available',
            )
            property.save()

            for image in property_images:
                property_image = PropertyImage(property=property, image=image)
                property_image.save()
            messages.success(request, 'Property list is pending to approve from admin!')
            return redirect('seller_view')
        except ValidationError as e:
            messages.error(request, e.message)
        except Exception as e:
            messages.error(request, 'Error saving property: ' + str(e))

    return render(request, 'seller.html')

@login_required
def update_sale_status(request, property_id):
    if request.method == 'POST':
        property = get_object_or_404(Property, id=property_id, seller=request.user)
        new_status = request.POST.get('sale_status')
        if new_status in dict(Property.SALE_STATUS_CHOICES):
            property.sale_status = new_status
            property.save()
            messages.success(request, 'Sale status updated successfully!')
        else:
            messages.error(request, 'Invalid sale status!')
    return redirect('user_dashboard')

def property_detail(request, property_id):
    property_obj = get_object_or_404(Property, id=property_id)
    seller_email = property_obj.seller.email
    recommended_properties = recommend_similar_properties(property_obj)

    error_message = None  # default error message

    if request.method == "POST":
        sender_name = request.POST.get("sender_name")
        sender_email = request.POST.get("sender_email")
        content = request.POST.get("content")

        if sender_name and sender_email and content:
            message = Message(
                sender_name=sender_name,
                sender_email=sender_email,
                content=content,
                property=property_obj,
            )
            message.save()

            # Email to seller
            seller_subject = f"Inquiry about {property_obj.title}"
            seller_message = render_to_string('email/buyer_seller_communication.html', {
                'recipient_name': property_obj.seller.first_name,
                'is_seller': True,
                'sender_name': sender_name,
                'sender_email': sender_email,
                'message': content,
                'property': property_obj,
            })

            seller_email_msg = EmailMessage(
                subject=seller_subject,
                body=seller_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[seller_email],
                reply_to=[sender_email],
            )
            seller_email_msg.send(fail_silently=False)

            # Confirmation email to buyer
            buyer_subject = f"Confirmation: Inquiry about {property_obj.title}"
            buyer_message = render_to_string('email/buyer_seller_communication.html', {
                'recipient_name': sender_name,
                'is_seller': False,
                'sender_name': sender_name,
                'sender_email': sender_email,
                'message': content,
                'property': property_obj,
            })

            buyer_email_msg = EmailMessage(
                subject=buyer_subject,
                body=buyer_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[sender_email],
            )
            buyer_email_msg.send(fail_silently=False)

            messages.success(request, "Your message has been sent to the seller.")
            return redirect("property_detail", property_id=property_id)
        else:
            error_message = "Please fill in all fields."

    return render(
        request,
        "property_detail.html",
        {
            "property": property_obj,
            "recommended_properties": recommended_properties,
            "error_message": error_message,
        },
    )
def edit_property(request, property_id):
    # Allow superusers to edit any property, otherwise only the seller
    if request.user.is_superuser:
        property_obj = get_object_or_404(Property, id=property_id)
    else:
        property_obj = get_object_or_404(Property, id=property_id, seller=request.user)
    
    if request.method == 'POST':        
        # Convert boolean fields
        boolean_fields = [
            'mainroad', 'guestroom', 'basement', 
            'hotwaterheating', 'airconditioning'
        ]
        
        try:
            # Update basic fields
            property_obj.title = request.POST.get('title', '').strip()
            property_obj.city = request.POST.get('city')
            property_obj.area = request.POST.get('area')
            property_obj.bedrooms = request.POST.get('bedrooms')
            property_obj.bathrooms = request.POST.get('bathrooms')
            property_obj.stories = request.POST.get('stories')
            property_obj.parking = request.POST.get('parking')
            property_obj.furnishingstatus = request.POST.get('furnishingstatus')
            property_obj.price = request.POST.get('price')
            property_obj.sale_status = request.POST.get('sale_status')
            # Handle boolean fields
            for field in boolean_fields:
                # Convert string 'True'/'False' to actual boolean
                value = request.POST.get(field)
                if value == 'True':
                    setattr(property_obj, field, True)
                elif value == 'False':
                    setattr(property_obj, field, False)
            
            # Handle image upload (optional)
            if 'property_image' in request.FILES:
                property_obj.property_image = request.FILES['property_image']
            
            # Save the updated property
            property_obj.save()
            
            messages.success(request, f"Property '{property_obj.title}' updated successfully!")
            return redirect('user_dashboard')
        
        except Exception as e:
            # Log the error for debugging
            print(f"Error updating property: {e}")
            messages.error(request, "An error occurred while updating the property.")
    
    # For GET request, render the edit page
    return render(request, 'edit_property.html', {
        'property': property_obj,
        'city_choices': Property.CITY_CHOICES,
        'furnishing_choices': Property.FURNISHING_STATUS_CHOICES,
        'sale_status_choices':Property.SALE_STATUS_CHOICES
    })

def delete_property(request, property_id):
    property = get_object_or_404(Property, id=property_id, seller=request.user)

    if request.method == 'POST':
        property.delete()
        messages.success(request, f"Property '{property.title}' deleted successfully!")
        return redirect('user_dashboard')

    return render(request, 'confirm_delete.html', {'property': property})


@login_required
def predict(request):
    try:
        feature_scaler = pickle.load(open('house/ml_models/saved_models/feature_scaler.pkl', 'rb'))
        svm_model = pickle.load(open('house/ml_models/saved_models/svm_model.pkl', 'rb'))
        dt_model = pickle.load(open('house/ml_models/saved_models/decision_tree.pkl', 'rb'))
        
        with open('house/ml_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        if request.method == 'POST':
            try:
                area = float(request.POST.get('area', 0))
                stories = float(request.POST.get('stories', 0))
                road_width = float(request.POST.get('road_width', 0))
                
                if area <= 0 or stories <= 0 or road_width <= 0:
                    raise ValueError("Input values must be positive numbers")
                
            except ValueError as e:
                return render(request, 'predict.html', {'error': str(e)})
            
            city = request.POST.get('city')
            road_type = request.POST.get('road_type')

            # Create initial DataFrame from raw input
            input_data = {
                'Floors': stories,
                'Area': area,
                'Road_Width': road_width,
                'City_Bhaktapur': 1 if city == 'Bhaktapur' else 0,
                'City_Kathmandu': 1 if city == 'Kathmandu' else 0,
                'City_Lalitpur': 1 if city == 'Lalitpur' else 0,
                'Road_Type_Blacktopped': 1 if road_type == 'Blacktopped' or road_type == 'Paved' or road_type == 'Concrete' else 0,
                'Road_Type_Gravelled': 1 if road_type == 'Gravelled' else 0,
                'Road_Type_Soil Stabilized': 1 if road_type == 'Soil_Stabilized' else 0,
                'Price': 0.0 # Add a dummy price column for feature engineering function
            }
            
            df = pd.DataFrame([input_data])

            # Apply the same feature engineering as in model_train.py
            df_engineered = engineer_features(df)

            # Select only the features the model was trained on
            df_processed = df_engineered.reindex(columns=feature_names, fill_value=0)

            # Handle potential inf/-inf/NaN introduced by feature engineering
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    # For simplicity, fill NaNs with 0 for prediction input; consider a more robust strategy if needed
                    df_processed[col] = df_processed[col].fillna(0)
            

            X_scaled = feature_scaler.transform(df_processed)
            

            svm_pred_log = svm_model.predict(X_scaled)[0]
            dt_pred_log = dt_model.predict(X_scaled)[0]
            

            svm_pred_price = np.expm1(svm_pred_log)
            dt_pred_price = np.expm1(dt_pred_log)
            
            # Apply reasonable price thresholds for Nepal real estate market
            # MIN_PRICE = 20000000  # 1 million NPR
            # MAX_PRICE = 500000000  # 500 million NPR
            
            svm_pred_price = np.clip(svm_pred_price)
            dt_pred_price = np.clip(dt_pred_price)
            
            try:
                prediction = HousePricePrediction.objects.create(
                    area=area,
                    stories=stories,
                    road_width=road_width,
                    city=city,
                    road_type=road_type,
                    svm_prediction=svm_pred_price,
                    dt_prediction=dt_pred_price
                )
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
            
            return render(request, 'predict.html', {
                'prediction_svm': f'Rs. {svm_pred_price:,.2f}',
                'prediction_dt': f'Rs. {dt_pred_price:,.2f}',
                'form_data': request.POST,
                'input_area': area,
                'input_stories': stories,
                'input_road_width': road_width,
                'input_city': city,
                'input_road_type': road_type
            })
        
        return render(request, 'predict.html')
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render(request, 'predict.html', {
            'error': f"An error occurred: {str(e)}"
        })
    
def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        if name and email and subject and message:
            ContactMessage.objects.create(
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            messages.success(request, "Your message has been sent successfully!")
            return redirect('contact') 
        else:
            messages.error(request, "Please fill in all fields.")

    return render(request, 'contact.html')

@user_passes_test(lambda u: u.is_superuser)
def pending_approvals(request):
    """View pending property approvals"""
    properties = Property.objects.filter(is_approved=False, decline_reason__isnull=True)
    return render(request, 'pending_approvals.html', {'properties': properties})

@user_passes_test(lambda u: u.is_superuser)
def manage_users(request):
    """Manage user accounts"""
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'manage_users.html', {'users': users})

@user_passes_test(lambda u: u.is_superuser)
def manage_properties(request):
    """Manage all properties"""
    properties = Property.objects.all().order_by('-created_at')
    return render(request, 'manage_properties.html', {'properties': properties})

@user_passes_test(lambda u: u.is_superuser)
def view_messages(request):
    """View and manage user messages"""
    user_messages = Message.objects.all().order_by('-sent_at')
    return render(request, 'view_messages.html', {'user_messages': user_messages})

@user_passes_test(lambda u: u.is_superuser)
def edit_user(request, user_id):
    """Edit user details"""
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        try:
            user.username = request.POST.get('username')
            user.email = request.POST.get('email')
            user.first_name = request.POST.get('first_name')
            user.last_name = request.POST.get('last_name')
            user.save()
            messages.success(request, f"User {user.username} updated successfully!")
        except Exception as e:
            messages.error(request, f"Error updating user: {str(e)}")
        return redirect('manage_users')
    return redirect('manage_users')

@user_passes_test(lambda u: u.is_superuser)
def activate_user(request, user_id):
    """Activate a user account"""
    user = get_object_or_404(User, id=user_id)
    user.is_active = True
    user.save()
    messages.success(request, f"User {user.username} activated successfully!")
    return redirect('manage_users')

@user_passes_test(lambda u: u.is_superuser)
def deactivate_user(request, user_id):
    """Deactivate a user account"""
    user = get_object_or_404(User, id=user_id)
    user.is_active = False
    user.save()
    messages.success(request, f"User {user.username} deactivated successfully!")
    return redirect('manage_users')

@user_passes_test(lambda u: u.is_superuser)
def delete_user(request, user_id):
    """Delete a user account"""
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        username = user.username
        user.delete()
        messages.success(request, f"User {username} deleted successfully!")
    return redirect('manage_users')

@user_passes_test(lambda u: u.is_superuser)
def unapprove_property(request, property_id):
    """Unapprove a property"""
    property = get_object_or_404(Property, id=property_id)
    property.is_approved = False
    property.save()
    messages.success(request, f"Property '{property.title}' unapproved successfully!")
    return redirect('manage_properties')

@user_passes_test(lambda u: u.is_superuser)
def reply_message(request, message_id):
    """Reply to a user message"""
    message = get_object_or_404(Message, id=message_id)
    if request.method == 'POST':
        try:
            subject = request.POST.get('subject')
            reply_content = request.POST.get('message')
            recipient_email = request.POST.get('recipient')

            # Send email reply
            email = EmailMessage(
                subject=subject,
                body=reply_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[recipient_email],
                reply_to=[settings.DEFAULT_FROM_EMAIL]
            )
            email.send(fail_silently=False)

            # Mark original message as read
            message.is_read = True
            message.save()

            messages.success(request, "Reply sent successfully!")
        except Exception as e:
            messages.error(request, f"Error sending reply: {str(e)}")
    return redirect('view_messages')

@user_passes_test(lambda u: u.is_superuser)
def delete_message(request, message_id):
    """Delete a user message"""
    message = get_object_or_404(Message, id=message_id)
    if request.method == 'POST':
        message.delete()
        messages.success(request, "Message deleted successfully!")
    return redirect('view_messages')

@user_passes_test(lambda u: u.is_superuser)
def add_user(request):
    """Add a new user"""
    if request.method == 'POST':
        try:
            username = request.POST.get('username')
            email = request.POST.get('email')
            full_name = request.POST.get('full_name')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            user_type = request.POST.get('user_type')

            # Validate passwords match
            if password != confirm_password:
                messages.error(request, "Passwords do not match.")
                return redirect('manage_users')

            # Validate password strength
            is_valid, error_message = validate_password(password)
            if not is_valid:
                messages.error(request, error_message)
                return redirect('manage_users')

            # Check if email already exists
            if User.objects.filter(email=email).exists():
                messages.error(request, "Email already in use.")
                return redirect('manage_users')

            # Check if username already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already in use.")
                return redirect('manage_users')

            # Split full name into first and last name
            name_parts = full_name.split()
            first_name = name_parts[0]
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''

            # Create the user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                is_active=True
            )

            # Set user type (you might want to add a user_type field to your User model)
            if user_type == 'seller':
                user.is_staff = True  # or however you distinguish sellers

            user.save()
            messages.success(request, f"User {username} created successfully!")
            return redirect('manage_users')

        except Exception as e:
            messages.error(request, f"Error creating user: {str(e)}")
            return redirect('manage_users')

    return redirect('manage_users')

@user_passes_test(lambda u: u.is_superuser)
def view_contact_messages(request):
    """View and manage contact messages"""
    contact_messages = ContactMessage.objects.all().order_by('-created_at')
    return render(request, 'view_contact_messages.html', {'contact_messages': contact_messages})

@login_required
def add_to_wishlist(request, property_id):
    property = get_object_or_404(Property, id=property_id)
    Wishlist.objects.get_or_create(user=request.user, property=property)
    return redirect('property_detail', property_id=property_id)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from house.models import Property

def recommend_similar_properties(target_property, top_n=4, area_tolerance=0.5, price_tolerance=0.5):
    # Only consider properties in the same city
    queryset = Property.objects.filter(
        is_approved=True,
        city=target_property.city
    ).exclude(id=target_property.id)

    if not queryset.exists():
        return []
    
    #TO recommend on basis of area and price

    # min_area = float(target_property.area) * (1 - area_tolerance)
    # max_area = float(target_property.area) * (1 + area_tolerance)
    # min_price = float(target_property.price) * (1 - price_tolerance)
    # max_price = float(target_property.price) * (1 + price_tolerance)

    # queryset = queryset.filter(
    #     area__gte=min_area, area__lte=max_area,
    #     price__gte=min_price, price__lte=max_price
    # )

    if not queryset.exists():
        return []

    # Build DataFrame for area and price
    df = pd.DataFrame(list(queryset.values('id', 'area', 'price')))

    # Add the target property as the first row
    df = pd.concat([
        pd.DataFrame([{
            'id': target_property.id,
            'area': float(target_property.area),
            'price': float(target_property.price),
        }]),
        df
    ], ignore_index=True)

    # Scale area and price
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = ['area', 'price']
    df_scaled = scaler.fit_transform(df[features])

    # Compute cosine similarity (first row is the target)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([df_scaled[0]], df_scaled[1:])[0]

    # Sort by similarity, get top_n
    sorted_indices = similarities.argsort()[::-1][:top_n]

    # Get the property IDs of the most similar properties
    similar_ids = df.iloc[[i+1 for i in sorted_indices]]['id'].tolist()

    # Preserve order using Case/When for the queryset
    from django.db.models import Case, When
    preserved = Case(*[When(id=pk, then=pos) for pos, pk in enumerate(similar_ids)])
    similar_properties = Property.objects.filter(id__in=similar_ids).order_by(preserved)

    return similar_properties