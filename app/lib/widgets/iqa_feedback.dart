import 'package:flutter/material.dart';

import '../config/theme.dart';
import '../services/camera_service.dart';

/// Displays image-quality feedback after a capture.
///
/// Shows a green checkmark for good images and a red/amber warning
/// with a short explanation for poor-quality captures.
class IqaFeedbackWidget extends StatelessWidget {
  final ImageQuality quality;

  const IqaFeedbackWidget({super.key, required this.quality});

  @override
  Widget build(BuildContext context) {
    final config = _configFor(quality);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeOut,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: config.backgroundColor,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: config.iconColor.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(config.icon, color: config.iconColor, size: 28),
          const SizedBox(width: 12),
          Flexible(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  config.title,
                  style: TextStyle(
                    color: config.iconColor,
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  config.subtitle,
                  style: TextStyle(
                    color: config.iconColor.withOpacity(0.85),
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  _FeedbackConfig _configFor(ImageQuality q) {
    switch (q) {
      case ImageQuality.good:
        return _FeedbackConfig(
          icon: Icons.check_circle_rounded,
          iconColor: NetraTheme.riskLow,
          backgroundColor: const Color(0xFFE8F8F0),
          title: 'Good Quality',
          subtitle: 'Image is clear and well-lit.',
        );
      case ImageQuality.blurry:
        return _FeedbackConfig(
          icon: Icons.blur_on_rounded,
          iconColor: NetraTheme.riskModerate,
          backgroundColor: const Color(0xFFFEF5E7),
          title: 'Possibly Blurry',
          subtitle: 'Try holding the device steadier.',
        );
      case ImageQuality.tooDark:
        return _FeedbackConfig(
          icon: Icons.brightness_low_rounded,
          iconColor: NetraTheme.riskHigh,
          backgroundColor: const Color(0xFFFDEDED),
          title: 'Too Dark',
          subtitle: 'Increase lighting and retake.',
        );
      case ImageQuality.tooLight:
        return _FeedbackConfig(
          icon: Icons.brightness_high_rounded,
          iconColor: NetraTheme.riskModerate,
          backgroundColor: const Color(0xFFFEF5E7),
          title: 'Overexposed',
          subtitle: 'Reduce lighting and retake.',
        );
      case ImageQuality.unknown:
        return _FeedbackConfig(
          icon: Icons.help_outline_rounded,
          iconColor: NetraTheme.textSecondary,
          backgroundColor: const Color(0xFFF2F3F4),
          title: 'Unknown Quality',
          subtitle: 'Could not assess image quality.',
        );
    }
  }
}

class _FeedbackConfig {
  final IconData icon;
  final Color iconColor;
  final Color backgroundColor;
  final String title;
  final String subtitle;

  const _FeedbackConfig({
    required this.icon,
    required this.iconColor,
    required this.backgroundColor,
    required this.title,
    required this.subtitle,
  });
}
