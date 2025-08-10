import requests
import markdown

# ====== CONFIG ======
PATREON_ACCESS_TOKEN = "rKBIoqnNOyzDVKFqDmNcXBiDpF6Yn7U0RkynJWFEWPw"  # Store token as env variable for safety
PATREON_CAMPAIGN_ID = "9291193"  # You can get this from Patreon API
# ====================

html_content = markdown.markdown("# Hello Patreon!")


def post_to_patreon(title, content_text, tier_ids=None, image_path=None, file_path=None):
    """
    Post content to Patreon.
    :param title: Post title
    :param content_text: Main body (supports HTML)
    :param tier_ids: List of tier IDs to lock content for
    :param image_path: Optional image to upload
    :param file_path: Optional file to attach
    """
    
    url = "https://www.patreon.com/api/oauth2/v2/posts"
    headers = {"Authorization": f"Bearer {PATREON_ACCESS_TOKEN}"}

    payload = {
    "data": {
        "type": "post",
        "attributes": {
            "title": "Daily Post",
            "content": html_content,
            "published": True
        },
        "relationships": {
            "campaign": {
                "data": {
                    "type": "campaign",
                    "id": PATREON_CAMPAIGN_ID
                }
            }
        }
    }
}

    # Step 1: Create the post
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 201:
        print("❌ Error creating post:", response.text)
        return None

    post_id = response.json()["data"]["id"]
    print(f"✅ Post created with ID: {post_id}")

    # Step 2: Attach image if provided
    if image_path:
        files = {"file": open(image_path, "rb")}
        img_url = f"https://www.patreon.com/api/oauth2/v2/posts/{post_id}/images"
        img_resp = requests.post(img_url, headers={"Authorization": f"Bearer {PATREON_ACCESS_TOKEN}"}, files=files)
        if img_resp.status_code == 201:
            print("📸 Image uploaded successfully")
        else:
            print("❌ Error uploading image:", img_resp.text)

    # Step 3: Attach file if provided
    if file_path:
        files = {"file": open(file_path, "rb")}
        file_url = f"https://www.patreon.com/api/oauth2/v2/posts/{post_id}/attachments"
        file_resp = requests.post(file_url, headers={"Authorization": f"Bearer {PATREON_ACCESS_TOKEN}"}, files=files)
        if file_resp.status_code == 201:
            print("📂 File uploaded successfully")
        else:
            print("❌ Error uploading file:", file_resp.text)

    return post_id


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    post_to_patreon(
        title="Today's Top Predictions",
        content_text="<p>Here are today's top predictions for our VIP members!</p>",
        tier_ids=["YOUR_TIER_ID"],  # e.g., VIP Tier
        image_path="tier2_premium_09-08-2025.png",
        file_path="tier3_vip_09-08-2025.csv"
    )
