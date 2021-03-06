package pl.aprilapps.easyphotopicker

/**
 * Stas Parshin
 * 05 November 2015
 */
abstract class DefaultCallback : RespiratoryDisease.Callbacks {

    override fun onImagePickerError(error: Throwable, source: MediaSource) {}

    override fun onCanceled(source: MediaSource) {}
}
