# permissions.py
from rest_framework.permissions import BasePermission
from django.conf import settings

class HasAccessCode(BasePermission):
    """
    Custom permission to only allow access to users who provide a specific access code.
    """
    def has_permission(self, request, view):
        access_code = request.headers.get('X-Token-Code')
        return access_code == settings.FIXED_ACCESS_CODE
