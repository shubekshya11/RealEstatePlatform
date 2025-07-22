from django.urls import path
from house.views import *
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('activate/<uidb64>/<token>/', activate_view, name='activate'),
    path('password-reset/', password_reset_view, name='password_reset'),
    path('password-reset-confirm/<uidb64>/<token>/', password_reset_confirm_view, name='password_reset_confirm_view'),
    path('', home, name='home'),
    path("buyer/", buyer, name='buyer'),
    path("seller/", seller_view, name='seller_view'),
    path('property/<int:property_id>/', property_detail, name='property_detail'),
    path("predict/",predict, name="predict"),
    path("contact/",contact, name='contact'), 
    path("buyer/pending/", pending_properties, name="pending_properties"),
    path("buyer/approve/<int:property_id>/", approve_property, name="approve_property"),
    path('decline_property/<int:property_id>/', decline_property, name='decline_property'),
    path("dashboard/", user_dashboard, name="user_dashboard"),
    path("admin-dashboard/", admin_dashboard, name="admin_dashboard"),
    path('view-dataset/<str:filename>/', view_dataset, name='view_dataset'),
    path('property/edit/<int:property_id>/', edit_property, name='edit_property'),
    path('property/delete/<int:property_id>/', delete_property, name='delete_property'),
    path('property/<int:property_id>/save/', add_to_wishlist, name='add_to_wishlist'),
    
    # Admin dashboard URLs with 'dashboard' prefix instead of 'admin'
    path('dashboard/pending-approvals/', pending_approvals, name='pending_approvals'),
    path('dashboard/manage-users/', manage_users, name='manage_users'),
    path('dashboard/manage-properties/', manage_properties, name='manage_properties'),
    path('dashboard/view-messages/', view_messages, name='view_messages'),
    path('dashboard/edit-user/<int:user_id>/', edit_user, name='edit_user'),
    path('dashboard/activate-user/<int:user_id>/', activate_user, name='activate_user'),
    path('dashboard/deactivate-user/<int:user_id>/', deactivate_user, name='deactivate_user'),
    path('dashboard/delete-user/<int:user_id>/', delete_user, name='delete_user'),
    path('dashboard/add-user/', add_user, name='add_user'),
    path('dashboard/unapprove-property/<int:property_id>/', unapprove_property, name='unapprove_property'),
    path('dashboard/reply-message/<int:message_id>/', reply_message, name='reply_message'),
    path('dashboard/delete-message/<int:message_id>/', delete_message, name='delete_message'),
    path('dashboard/view-contact-messages/', view_contact_messages, name='view_contact_messages'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)