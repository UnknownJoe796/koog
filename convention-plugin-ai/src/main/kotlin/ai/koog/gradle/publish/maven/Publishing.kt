package ai.koog.gradle.publish.maven

import org.gradle.api.Action
import org.gradle.api.Project
import org.gradle.api.artifacts.dsl.RepositoryHandler
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import java.io.File
import java.net.URI
import java.net.URL

object Publishing {
    fun Project.publishToMaven() {
        publishTo({
            it.graziePublic(project)
            it.artifactsMaven(project)
            it.s3Maven(project)
        }) {
            it.publications(
                Action {
                    val publications = this

                    publications.forEach {
                        val p = it as MavenPublication
                        p.pom(
                            Action {
                                val pom = this

                                pom.name.set(this@publishToMaven.name)
                                pom.description.set("Koog is a framework for quickly creating AI agents in Kotlin with minimal effort.")
                                pom.url.set("https://github.com/JetBrains/koog")

                                pom.licenses(
                                    Action {
                                        val licenses = this

                                        licenses.license(
                                            Action {
                                                val license = this

                                                license.name.set("The Apache License, Version 2.0")
                                                license.url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                                            }
                                        )
                                    }
                                )

                                pom.developers(
                                    Action {
                                        val developers = this

                                        developers.developer(
                                            Action {
                                                val developer = this

                                                developer.id.set("JetBrains")
                                                developer.name.set("JetBrains Team")
                                                developer.organization.set("JetBrains")
                                                developer.organizationUrl.set("https://www.jetbrains.com")
                                            }
                                        )
                                    }
                                )

                                pom.scm(
                                    Action {
                                        val scm = this
                                        scm.url.set("https://github.com/JetBrains/koog.git")
                                    }
                                )
                            }
                        )
                    }
                }
            )
        }
    }

    private fun Project.publishTo(
        configureRepository: (RepositoryHandler) -> Unit,
        configurePublish: (PublishingExtension) -> Unit = {}
    ) {
        pluginManager.apply("maven-publish")

        extensions.configure<PublishingExtension>("publishing") {
            repositories(Action { configureRepository(this) })
            configurePublish(this)
        }
    }

    private fun RepositoryHandler.artifactsMaven(project: Project) {
        maven(
            Action {
                val repo = this

                repo.name = "artifacts"
                repo.url = project.rootProject.layout.buildDirectory.dir("artifacts/maven").get().asFile.toURI()
            }
        )
    }

    private fun RepositoryHandler.graziePublic(project: Project) {
        maven(
            Action {
                val repo = this

                repo.name = "GraziePublicMaven"
                repo.url = URL("https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public").toURI()

                repo.credentials(
                    Action {
                        val cred = this

                        cred.username = project.properties["spaceUsername"]?.toString()
                            ?: System.getenv("JB_SPACE_CLIENT_ID")

                        cred.password = project.properties["spacePassword"]?.toString()
                            ?: System.getenv("JB_SPACE_CLIENT_SECRET")
                    }
                )
            }
        )
    }

    private fun RepositoryHandler.s3Maven(project: Project) {
        maven(
            Action {
                val repo = this

                repo.name = "S3Maven"
                repo.url = URI.create("s3://lightningkite-maven")

                // Check if explicit credentials are provided
                val explicitAccessKey = project.properties["awsAccessKeyId"]?.toString()
                    ?: System.getenv("AWS_ACCESS_KEY_ID")
                val explicitSecretKey = project.properties["awsSecretAccessKey"]?.toString()
                    ?: System.getenv("AWS_SECRET_ACCESS_KEY")

                if (explicitAccessKey != null && explicitSecretKey != null) {
                    // Use explicit credentials if provided
                    repo.credentials(org.gradle.api.credentials.AwsCredentials::class.java,
                        Action {
                            val cred = this
                            cred.accessKey = explicitAccessKey
                            cred.secretKey = explicitSecretKey

                            val sessionToken = project.properties["awsSessionToken"]?.toString()
                                ?: System.getenv("AWS_SESSION_TOKEN")
                            if (sessionToken != null) {
                                cred.sessionToken = sessionToken
                            }
                        }
                    )
                } else {
                    // Read credentials from AWS credentials file
                    val awsProfile = project.properties["awsProfile"]?.toString()
                        ?: System.getenv("AWS_PROFILE")
                        ?: "lk"

                    val credentials = readAwsCredentials(awsProfile)
                    if (credentials != null) {
                        repo.credentials(org.gradle.api.credentials.AwsCredentials::class.java,
                            Action {
                                val cred = this
                                cred.accessKey = credentials.accessKeyId
                                cred.secretKey = credentials.secretAccessKey
                                if (credentials.sessionToken != null) {
                                    cred.sessionToken = credentials.sessionToken
                                }
                            }
                        )
                    } else {
                        project.logger.warn("Could not find AWS credentials for profile '$awsProfile'")
                    }
                }
            }
        )
    }

    private data class AwsCredentials(
        val accessKeyId: String,
        val secretAccessKey: String,
        val sessionToken: String? = null
    )

    private fun readAwsCredentials(profileName: String): AwsCredentials? {
        val homeDir = System.getProperty("user.home")
        val credentialsFile = File(homeDir, ".aws/credentials")

        if (!credentialsFile.exists()) {
            return null
        }

        var inProfile = false
        var accessKeyId: String? = null
        var secretAccessKey: String? = null
        var sessionToken: String? = null

        credentialsFile.readLines().forEach { line ->
            val trimmedLine = line.trim()

            when {
                trimmedLine.startsWith("[") && trimmedLine.endsWith("]") -> {
                    val currentProfile = trimmedLine.substring(1, trimmedLine.length - 1)
                    inProfile = currentProfile == profileName
                }
                inProfile && trimmedLine.startsWith("aws_access_key_id") -> {
                    accessKeyId = trimmedLine.substringAfter("=").trim()
                }
                inProfile && trimmedLine.startsWith("aws_secret_access_key") -> {
                    secretAccessKey = trimmedLine.substringAfter("=").trim()
                }
                inProfile && trimmedLine.startsWith("aws_session_token") -> {
                    sessionToken = trimmedLine.substringAfter("=").trim()
                }
            }
        }

        return if (accessKeyId != null && secretAccessKey != null) {
            AwsCredentials(accessKeyId!!, secretAccessKey!!, sessionToken)
        } else {
            null
        }
    }
}
