from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image and not image.content_type.startswith('image/'):
            raise forms.ValidationError('File must be an image.')
        return image