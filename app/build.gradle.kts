plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

val releaseVersionCode = providers.environmentVariable("KIRAENHANCE_VERSION_CODE")
    .orNull
    ?.toIntOrNull()
val releaseVersionName = providers.environmentVariable("KIRAENHANCE_VERSION_NAME")
    .orNull
    ?.takeIf { it.isNotBlank() }

android {
    namespace = "com.ikegami99.kiraenhance"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.ikegami99.kiraenhance"
        minSdk = 28
        targetSdk = 36
        versionCode = releaseVersionCode ?: 1
        versionName = releaseVersionName ?: "0.1.0-alpha01"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                providers.environmentVariable("NCNN_DIR").orNull
                    ?.takeIf { it.isNotBlank() }
                    ?.let { arguments += "-Dncnn_DIR=$it" }
                providers.environmentVariable("MNN_INCLUDE_DIR").orNull
                    ?.takeIf { it.isNotBlank() }
                    ?.let { arguments += "-DMNN_INCLUDE_DIR=$it" }
                providers.environmentVariable("MNN_LIB_DIR").orNull
                    ?.takeIf { it.isNotBlank() }
                    ?.let { arguments += "-DMNN_LIB_DIR=$it" }
            }
        }
    }

    buildFeatures {
        compose = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.06.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.10.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.10.0")
    implementation("androidx.navigation:navigation-compose:2.9.8")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.work:work-runtime:2.11.2")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.2")
    implementation("com.squareup.okhttp3:okhttp:5.3.2")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
}
