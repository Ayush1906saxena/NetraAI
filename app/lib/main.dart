import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:go_router/go_router.dart';

import 'config/theme.dart';
import 'screens/home_screen.dart';
import 'screens/patient_registration.dart';
import 'screens/camera_capture.dart';
import 'screens/processing_screen.dart';
import 'screens/results_screen.dart';
import 'screens/history_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait mode for consistent medical UI
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Initialize Hive for local storage and offline queue
  await Hive.initFlutter();
  await Hive.openBox('offlineQueue');
  await Hive.openBox('screeningsCache');
  await Hive.openBox('settings');

  runApp(
    const ProviderScope(
      child: NetraApp(),
    ),
  );
}

/// GoRouter configuration for the app navigation
final _router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      name: 'home',
      builder: (context, state) => const HomeScreen(),
    ),
    GoRoute(
      path: '/register',
      name: 'register',
      builder: (context, state) => const PatientRegistrationScreen(),
    ),
    GoRoute(
      path: '/capture/:screeningId',
      name: 'capture',
      builder: (context, state) {
        final screeningId = state.pathParameters['screeningId']!;
        return CameraCaptureScreen(screeningId: screeningId);
      },
    ),
    GoRoute(
      path: '/processing/:screeningId',
      name: 'processing',
      builder: (context, state) {
        final screeningId = state.pathParameters['screeningId']!;
        return ProcessingScreen(screeningId: screeningId);
      },
    ),
    GoRoute(
      path: '/results/:screeningId',
      name: 'results',
      builder: (context, state) {
        final screeningId = state.pathParameters['screeningId']!;
        return ResultsScreen(screeningId: screeningId);
      },
    ),
    GoRoute(
      path: '/history',
      name: 'history',
      builder: (context, state) => const HistoryScreen(),
    ),
  ],
);

class NetraApp extends StatelessWidget {
  const NetraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Netra AI',
      debugShowCheckedModeBanner: false,
      theme: NetraTheme.lightTheme,
      routerConfig: _router,
    );
  }
}
